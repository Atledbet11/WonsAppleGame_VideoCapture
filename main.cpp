// main.cpp
// Startup-only scaffold for WonsAppleGame VideoCapture rebuild.
// Focus: clean environment probing, device selection, and capture open.
// Next steps (later): thread classes, CPU/CUDA halves implementing a shared interface.

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#ifdef _WIN32
#include <windows.h>
#endif

// Include CUDA headers guarded; safe even if OpenCV was built w/o CUDA (calls will throw).
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utility.hpp>

#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

#include "video_capture.hpp"
#include "video_display.hpp"
#include "apple_detection.hpp"
#include "utils.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::getline;
using std::regex;
using std::smatch;
using std::string;
using std::to_string;
using std::vector;

using namespace cv;

static std::filesystem::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::filesystem::path p = (n ? std::filesystem::path(buf) : std::filesystem::path());
    return p.parent_path();
#else
    // Fallback: current_path; you can improve this for Linux/macOS if needed
    return std::filesystem::current_path();
#endif
}

// ---------- Small data types ----------

struct VideoDevice {
	int indexInList = -1;   // index within our enumerated video devices
	string name;            // display name reported by FFmpeg/DirectShow
};

enum class ComputeBackend { CPU = 0, CUDA = 1 };

struct Environment {
	string        ocvVersion;
	bool          builtWithCUDA   = false;
	bool          builtWithcuDNN  = false;
	int           cudaDeviceCount = 0;  // runtime-visible device count (0 if CPU)
	ComputeBackend backend        = ComputeBackend::CPU;
	int           selectedCamera  = -1;
};

// ---------- Forward decls ----------
static void   printCUDAReport(Environment& env);
static void   ocvPrintBuildInfo();
static bool   ocvBuiltWithCUDA_heuristic();
static bool   ocvBuiltWithcuDNN_heuristic();

// ---------- ENTRY ----------
int main() {
	// Make OpenCV logger quieter (optional).
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

	// Temporary Debug:
	//cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
	// Windows:
	//_putenv_s("OPENCV_LOG_LEVEL", "INFO");

	cv::setNumThreads(1); // avoid CPU oversubscription during CUDA

	Environment env{};
	env.ocvVersion = CV_VERSION;

	printCUDAReport(env);
	ocvPrintBuildInfo(); // gated by DEBUG_OPENCV inside implementation

	// Decide backend: prefer CUDA only if (builtWithCUDA && cudaDeviceCount > 0)
	env.backend = (env.builtWithCUDA && env.cudaDeviceCount > 0)
				? ComputeBackend::CUDA
				: ComputeBackend::CPU;

	cout << "\nSelected backend: "
		<< (env.backend == ComputeBackend::CUDA ? "CUDA" : "CPU")
		<< "\n";

	// ---- Configure the video_capture wrapper ----
	video_capture cam;

	// Let the camera class handle selection UI
	int camIndex = cam.selectVideoDevice();
	if (camIndex < 0) {
		std::cerr << "No valid device selected. Exiting.\n";
		return -1;
	}
	env.selectedCamera = camIndex;

	cam.set_camera_index(env.selectedCamera);
	cam.set_backend(env.backend == ComputeBackend::CUDA
						? video_capture::Backend::CUDA
						: video_capture::Backend::CPU);

	// Request 1080p60, but let the class search smartly and verify:
	auto info = cam.negotiate_auto(/*target_w=*/1920, /*target_h=*/1080,
								/*min_fps=*/59.0,
								/*prefer_mjpg=*/true, /*measure_ms=*/1200);
	double fps_final = (info.fps_measured > 0.0) ? info.fps_measured : info.fps_reported;
	std::cout << "Selected: " << info.width << "x" << info.height
			<< " ~" << fps_final << " fps"
	#ifdef _WIN32
			<< " via " << (cam.opened_backend()==video_capture::CapBackend::MSMF ? "MSMF" : "DSHOW")
	#endif
			<< "\n";

	// Start capture thread
	if (!cam.start_capture()) {
		std::cerr << "Failed to start capture.\n";
		return -1;
	}

	// Apple_detection Pre-Process

	std::atomic<bool> quit{false};

	apple_detection::Env aenv;
	aenv.backend     = apple_detection::Backend::CUDA; // or CPU
	aenv.prefer_fp16 = true;

	apple_detection detector(aenv, /*input_size*/ {480,480});
	auto model_path = exe_dir() / "apples.onnx";
	if (!std::filesystem::exists(model_path)) {
		std::cerr << "ERROR: apples.onnx not found at: " << model_path.string() << "\n";
		std::cerr << "Place apples.onnx next to the executable, or update the path.\n";
		return -2;
	}
	std::cout << "Using model: " << model_path.string() << "\n";
	detector.set_model_path(model_path.string());

	detector.set_conf_threshold(0.25f);
	detector.set_nms_threshold(0.45f);

	detector.start();

	std::thread to_detector([&](){
		using namespace std::chrono_literals;
		cv::Mat snap;
		while (!quit.load()) {
			// Always give detector a consistent CPU frame
			if (cam.snapshot_gpu_download(snap) || cam.snapshot_cpu(snap)) {
				if (!snap.empty()) detector.push_cpu(snap);
			}
			std::this_thread::sleep_for(2ms);
		}
	});

	// ---- Display ----
	video_display disp;
	// disp.output_dir = "F:/ComputerScience/WonsAppleGame/output"; // (optional)
	disp.set_input_callback([&](int key){
		switch (key & 0xFF) {
			case 's': case 'S': disp.saveScreenshot(); break;
			case 'r': case 'R':
				if (!disp.isRecording()) disp.startRecording(60);
				else disp.stopRecording();
				break;
			case 27: case 'q': case 'Q': quit.store(true); break;
		}
	});
	disp.start("Wons Mixed Up Apples");

	// Seed a basic diagnostic until detector starts publishing
	disp.diagnostic = cv::format("%dx%d ~%.1f fps [%s]  —  detector: warming up...",
								info.width, info.height, info.fps_measured,
								(env.backend==ComputeBackend::CUDA ? "CUDA" : "CPU"));

	// detector → display
	std::thread to_display([&](){
	using namespace std::chrono_literals;
	while (!quit.load()) {
		cv::Mat annotated = detector.get_annotated_clone();  // <-- get a fresh copy each loop
		if (!annotated.empty()) {
		disp.setFrame(annotated);
		disp.diagnostic = detector.diagnostic();
		}
		std::this_thread::sleep_for(2ms);
	}
	});


	// ---- Main wait loop ----
	// Keeps the process from shutting down early.
	while (!quit.load()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	// ---- Shutdown ----
	if (to_detector.joinable()) to_detector.join();
	if (to_display.joinable())  to_display.join();

	disp.stopRecording();
	disp.stop();
	detector.stop();
	cam.stop_capture();
	return 0;
}

// ---------- IMPLEMENTATION ----------

static void printCUDAReport(Environment& env) {
	cout << "OpenCV version: " << env.ocvVersion << "\n";

	// Heuristics: parse build info + safe runtime calls
	env.builtWithCUDA  = ocvBuiltWithCUDA_heuristic();
	env.builtWithcuDNN = ocvBuiltWithcuDNN_heuristic();

	cout << "Built with CUDA?  "  << (env.builtWithCUDA  ? "YES" : "NO") << "\n";
	cout << "Built with cuDNN? " << (env.builtWithcuDNN ? "YES" : "NO") << "\n";

	try {
		env.cudaDeviceCount = cv::cuda::getCudaEnabledDeviceCount();
	} catch (const cv::Exception& e) {
		// Thrown if OpenCV not built with CUDA or CUDA runtime absent
		cout << "Unable to find cuda enabled device " << e.what() << "\n";
		env.cudaDeviceCount = 0;
	}
	cout << "CUDA devices visible: " << env.cudaDeviceCount << "\n";

	if (env.cudaDeviceCount > 0) {
		try {
			cv::cuda::DeviceInfo dev0(0);
			cout << "Device 0: " << dev0.name()
				<< ", CC " << dev0.majorVersion() << "." << dev0.minorVersion() << "\n";
		} catch (const cv::Exception& e) {
			cout << "CUDA device query failed: " << e.what() << "\n";
		}
	}

#ifdef DEBUG_CUDA
	// Optional: print full device info for device 0
	if (env.cudaDeviceCount > 0) {
		try {
			cv::cuda::printShortCudaDeviceInfo(0);
		} catch (const cv::Exception& e) {
			cout << "printShortCudaDeviceInfo failed: " << e.what() << "\n";
		}
	}
#endif
}

static void ocvPrintBuildInfo() {
#ifdef DEBUG_OPENCV
	string info = cv::getBuildInformation();
	cout << info << endl;
#else
	// No-op unless DEBUG_OPENCV is defined
#endif
}

// Very light heuristics that don’t crash CPU-only builds:
static bool ocvBuiltWithCUDA_heuristic() {
	// Parse build info (most reliable).
	try {
		const string info = cv::getBuildInformation();
		// OpenCV build info lines typically contain "CUDA: YES/NO"
		regex cudaRegex(R"(CUDA\s*:\s*YES)", regex::icase);
		return regex_search(info, cudaRegex);
	} catch (...) {
		// ignore
	}
	// Fallback: try a harmless runtime call
	try {
		(void)cv::cuda::getCudaEnabledDeviceCount();
		return true;
	} catch (...) {
		return false;
	}
}

static bool ocvBuiltWithcuDNN_heuristic() {
	try {
		const string info = cv::getBuildInformation();
		// cuDNN line varies by version; catch common variants
		// Regex: CUDNN*YES
		regex cudnnRegex(R"(CUDNN\s*:\s*YES)", regex::icase);
		return regex_search(info, cudnnRegex);
	} catch (...) {
		return false;
	}
}
