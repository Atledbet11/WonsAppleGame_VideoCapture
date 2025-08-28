#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <regex>
#include <vector>
#include <filesystem>
#include <algorithm> // For std::max
#include <numeric>
#include <limits>

#include <cstdlib> // _putenv

#define NOMINMAX
#include <windows.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

//#define DEBUG_DNN 1

namespace fs = std::filesystem;

// Trackbar names (keep consistent)
static const char* TB_LOW   = "Low Threshold";
static const char* TB_HIGH  = "High Threshold";
static const char* TB_KER   = "Kernel(3/5/7)";

// Model path
fs::path model = fs::current_path() / "best.onnx";
const int modelResolution = 1920;
const float confidence = 0.25f;
const float intersection = 0.25f;

// Display Modes
enum modes {
	DEFAULT,
	CANNY,
	APPLES,
	MODES_END
};

// Structure that caches available video device information
struct VideoDevice {
	int index;
	string name;
};

// Used to store the Canny Parameters
struct CannyUIContext {
	Mat* gray;				// Gray Frame
	Mat* canny;				// Canny Frame
	int* lowThreshold;
	int* highThreshold;
	int* kernelIndex;
	int* kernel;
	const char* window;		// Window Name
};

// Generates a timestamp for used in file creation
static std::string timestamp()
{
	using clock = std::chrono::system_clock;
	auto now = clock::now();

	// Split into whole seconds since epoch and fractional milliseconds
	auto ms_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
	auto sec_since_epoch = std::chrono::duration_cast<std::chrono::seconds>(ms_since_epoch);

	// Milliseconds = remainder of division
	auto ms = ms_since_epoch - sec_since_epoch;

	// Convert seconds-since-epoch to calendar time
	std::time_t t = sec_since_epoch.count();
	tm tstamp{};
	#ifdef _WIN32
	localtime_s(&tstamp, &t);
	#else
	localtime_r(&t, &tstamp);
	#endif

	// Format: YYYYmmdd_HHMMSS_mmm
	std::ostringstream oss;
	oss << std::put_time(&tstamp, "%Y%m%d_%H%M%S")
		<< '_' << std::setw(3) << std::setfill('0') << ms.count();

	return oss.str();
}

// Struct for the recorder
// Used to start/stop mp4/avi recording.
struct Recorder {
	VideoWriter writer;
	bool active = false;
	bool save_frames = false;
	fs::path out_path;

	// Try MP4 first; fall back to AVI if needed
	bool start(int width, int height, double fps, const fs::path& outdir, int mode)
	{
		if (active) return true;

		// If the frame lacks "rgb" we have to save it without color.
		bool isColor = true;

		// Switch incase we have future modes that use black/white output.
		switch (mode) {
			case CANNY:
				isColor = false;
				break;
		}

		// Defining the output types
		int fourcc_mp4v = VideoWriter::fourcc('m','p','4','v');
		int fourcc_xvid = VideoWriter::fourcc('X','V','I','D');

		// Try to open an mp4 writer
		out_path = outdir / ("recording_" + timestamp() + ".mp4");
		bool ok = writer.open(out_path.string(), fourcc_mp4v, (fps > 0 ? fps : 30.0),
			Size(width, height), isColor);

		// If for some reason that fails, try an AVI format.
		if (!ok) {
			// Fallback to AVI/XVID
			out_path = outdir / ("recording_" + timestamp() + ".avi");
			ok = writer.open(out_path.string(), fourcc_xvid, (fps > 0 ? fps : 30.0),
				Size(width, height), isColor);
		}

		// If all else fails, we abandon this attempt and log an error
		if (!ok) {
			cerr << "Failed to open VideoWriter for recording.\n";
			return false;
		}

		// We have a successful connection, and will output the output path.
		active = true;
		cout << "Recording started: " << out_path.string() << "\n";
		return true;
	}

	// Write the provided frame to the file.
	void write(const Mat& frame)
	{
		// Make sure we are still recording, and the frame is not empty
		if (active && !frame.empty()) {
			writer.write(frame);
		}
	}

	// End the recoirding
	void stop()
	{
		// Sanity check
		if (active) {
			// Close out the video writer
			writer.release();

			// Reset flags for the next recording.
			active = false;
			save_frames = false;

			// Output the recording stopped, and the file path.
			cout << "Recording stopped: " << out_path.string() << "\n";
		}
	}
};

struct Detection {
	cv::Rect box;
	float score;
	int class_id;
};

// Start Adding helper functions to help debug my model discrepancy
static std::string humanBytes(uintmax_t b) {
	std::ostringstream oss;
	oss << b << " bytes (" << std::fixed << std::setprecision(2) << (b / (1024.0 * 1024.0)) << " MB)";
	return oss.str();
}

static std::string matShapeToStr(const cv::dnn::MatShape& s) {
	std::ostringstream oss; oss << "[";
	for (size_t i = 0; i < s.size(); ++i) { oss << s[i]; if (i + 1 < s.size()) oss << "x"; }
	oss << "]";
	return oss.str();
}

static void printBlobInfo(const cv::Mat& blob, const std::string& label) {
#if DEBUG_DNN
	std::cout << label << " blob: dims=" << blob.dims << " shape=[";
	for (int i = 0; i < blob.dims; ++i) {
		std::cout << blob.size[i] << (i + 1 < blob.dims ? "x" : "");
	}
	std::cout << "], type=" << blob.type() << "\n";
#endif
}

static void printNetSummary(cv::dnn::Net& net) {
#if DEBUG_DNN
	auto layerNames = net.getLayerNames();
	std::cout << "\n--- Net summary ---\n";
	std::cout << "Total layers: " << layerNames.size() << "\n";

	try {
		// In OpenCV 4.5.5 this is std::vector<int>
		std::vector<int> outIds = net.getUnconnectedOutLayers();
		std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();

		std::cout << "Unconnected output layer IDs: ";
		for (size_t i = 0; i < outIds.size(); ++i) {
			std::cout << outIds[i] << (i + 1 < outIds.size() ? ", " : "");
		}
		std::cout << "\nOutput names: ";
		for (size_t i = 0; i < outNames.size(); ++i) {
			std::cout << outNames[i] << (i + 1 < outNames.size() ? ", " : "");
		}
		std::cout << "\n";
	} catch (...) {
		std::cout << "(Could not query unconnected outputs)\n";
	}


	// Per-layer brief
	for (int idx = 0; idx < (int)layerNames.size(); ++idx) {
		int layerId = idx + 1; // OpenCV layers are 1-based
		cv::Ptr<cv::dnn::Layer> L = net.getLayer(layerId);
		if (!L) continue;
		std::cout << "  [" << layerId << "] " << (L->type.empty()? "?" : L->type)
				<< "  name=\"" << (L->name.empty()? layerNames[idx] : L->name) << "\"";

		// Show internal blobs (weights/bias) if present
		if (!L->blobs.empty()) {
			std::cout << "  (blobs: " << L->blobs.size() << ")";
			for (size_t b = 0; b < L->blobs.size(); ++b) {
				const auto& m = L->blobs[b];
				std::cout << "  blob" << b << ": dims=" << m.dims << " [";
				for (int d = 0; d < m.dims; ++d) {
					std::cout << m.size[d] << (d+1<m.dims?"x":"");
				}
				std::cout << "]";
			}
		}
		std::cout << "\n";
	}
	std::cout << "--- End summary ---\n\n";
#endif
}

// Summarize YOLO-style output [1, C, N]
static void debugAnalyzeOut(const cv::Mat& out, int modelW, int modelH, float confThresh) {
#if DEBUG_DNN
	printBlobInfo(out, "Output blob");
	if (out.dims != 3 || out.size[0] != 1) {
		std::cout << "[DNN][WARN] Unexpected output dims (expected [1,C,N]).\n";
		return;
	}
	const int C = out.size[1];
	const int N = out.size[2];
	std::cout << "[DNN] Channels C=" << C << " anchors N=" << N << "\n";

	// Per-channel min/max
	for (int c = 0; c < C; ++c) {
		const float* pc = out.ptr<float>(0, c);
		float mn = std::numeric_limits<float>::infinity();
		float mx = -std::numeric_limits<float>::infinity();
		for (int i = 0; i < N; ++i) { float v = pc[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
		std::cout << "  ch[" << c << "] min=" << mn << " max=" << mx << "\n";
	}

	// Assume last channel is confidence when C==5
	const int confCh = C - 1;
	const float* conf = out.ptr<float>(0, confCh);

	// Find top 10 by confidence
	std::vector<int> idx(N);
	std::iota(idx.begin(), idx.end(), 0);
	std::partial_sort(idx.begin(), idx.begin() + std::min(10, N), idx.end(),
		[&](int a, int b){ return conf[a] > conf[b]; });

	int countOver = 0;
	for (int i = 0; i < N; ++i) if (conf[i] >= confThresh) ++countOver;
	std::cout << "[DNN] conf >= " << confThresh << " : " << countOver << " anchors\n";

	std::cout << "[DNN] Top anchors by confidence:\n";
	for (int k = 0; k < std::min(10, N); ++k) {
		int i = idx[k];
		float score = conf[i];
		std::cout << "  #" << k << " i=" << i << " conf=" << score;
		if (C >= 4) {
			float x = out.ptr<float>(0, 0)[i];
			float y = out.ptr<float>(0, 1)[i];
			float w = out.ptr<float>(0, 2)[i];
			float h = out.ptr<float>(0, 3)[i];
			std::cout << " xywh=(" << x << "," << y << "," << w << "," << h << ")";
		}
		std::cout << "\n";
	}
#endif
}

static void probeLayerShapes(cv::dnn::Net& net) {
	std::vector<cv::dnn::MatShape> candidates = {
		{1,3,640,640}, {1,3,416,416}, {1,3,320,320}, {1,3,480,640}, {1,3,720,1280}
	};

	// Prefer a real unconnected output layer id; fallback to last layer name → id
	int targetLayerId = -1;
	try {
		auto outIds = net.getUnconnectedOutLayers(); // std::vector<int>
		if (!outIds.empty()) targetLayerId = outIds.back();
	} catch (...) {}

	if (targetLayerId < 0) {
		auto names = net.getLayerNames();
		if (!names.empty()) {
			try { targetLayerId = net.getLayerId(names.back()); } catch (...) {}
		}
	}
	if (targetLayerId < 0) {
		std::cout << "(probeLayerShapes) No valid target layer id found; skipping probes.\n\n";
		return;
	}

	std::cout << "--- Shape propagation probes (best-effort) ---\n";
	for (const auto& inShape : candidates) {
		try {
			std::vector<cv::dnn::MatShape> inShapes, outShapes;
			net.getLayerShapes(inShape, targetLayerId, inShapes, outShapes);
			std::cout << "Input " << matShapeToStr(inShape) << " -> inShapes("
					<< inShapes.size() << "), outShapes(" << outShapes.size() << ")\n";
			if (!outShapes.empty()) {
				std::cout << "  Example out shape: " << matShapeToStr(outShapes.back()) << "\n";
			}
		} catch (const cv::Exception& e) {
			std::cout << "Input " << matShapeToStr(inShape) << " -> shape propagation error: "
					<< e.err << " | " << e.func << " | " << e.msg << "\n";
		}
	}
	std::cout << "--- End shape probes ---\n\n";
}

// End Adding helper functions to help debug my model discrepancy
static cv::Mat letterbox_reshape(const cv::Mat& src, int new_w, int new_h,
                                 float& scale, int& dx, int& dy,
                                 const cv::Scalar& pad = cv::Scalar(114,114,114))
{
    const float r = std::min(new_w / (float)src.cols, new_h / (float)src.rows);
    const int nw = std::round(src.cols * r), nh = std::round(src.rows * r);
    scale = r;

    cv::Mat resized; cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    dx = (new_w - nw) / 2;  dy = (new_h - nh) / 2;

    cv::Mat out(new_h, new_w, src.type(), pad);
    resized.copyTo(out(cv::Rect(dx, dy, nw, nh)));
    return out;
}

// Letterbox to square with padding=114 (Ultralytics default)
static cv::Mat letterbox(const cv::Mat& src, int new_size, float& scale, int& dx, int& dy) {
	int w = src.cols, h = src.rows;
	float r = std::min((float)new_size / (float)w, (float)new_size / (float)h);
	int nw = std::round(w * r), nh = std::round(h * r);
	cv::Mat resized; 
	cv::resize(src, resized, cv::Size(nw, nh));

	cv::Mat out(new_size, new_size, src.type(), cv::Scalar(114,114,114));
	dx = (new_size - nw) / 2;
	dy = (new_size - nh) / 2;
	resized.copyTo(out(cv::Rect(dx, dy, nw, nh)));
	scale = r;
	return out;
}

// Parse Ultralytics YOLOv8/11 ONNX outputs (robust to (1,84,N) or (1,N,85))
// Assumptions:
//  - First 4 values are xywh (center-based)
//  - If an "objectness" exists, it's right after xywh (i.e., 5th value). Otherwise, class scores start at 5th.
//  - Remaining values are class scores (sigmoid-applied in export).
std::vector<Detection> parseDetections(const cv::Mat& out, float confThresh, float iouThresh, int /*inputSize_unused*/, int imgW, int imgH, float scale, int dx, int dy)
{
	std::vector<Detection> dets;

#if DEBUG_DNN
	std::cout << "[PARSE] enter: confThr=" << confThresh << " iouThr=" << iouThresh
			<< " img=(" << imgW << "x" << imgH << ") scale=" << scale
			<< " dx=" << dx << " dy=" << dy << "\n";
	printBlobInfo(out, "PARSE input");
#endif

	if (out.dims != 3 || out.size[0] != 1) {
#if DEBUG_DNN
		std::cout << "[PARSE][ERR] Unexpected output dims.\n";
#endif
		return dets;
	}

	const int C = out.size[1];
	const int N = out.size[2];
	const int confCh = C - 1;

	const float* X = out.ptr<float>(0, 0);
	const float* Y = (C > 1) ? out.ptr<float>(0, 1) : nullptr;
	const float* W = (C > 2) ? out.ptr<float>(0, 2) : nullptr;
	const float* H = (C > 3) ? out.ptr<float>(0, 3) : nullptr;
	const float* S = out.ptr<float>(0, confCh);

#if DEBUG_DNN
	int prelim = 0;
	float maxConf = 0.f; int maxIdx = -1;
	for (int i = 0; i < N; ++i) {
		if (S[i] >= confThresh) ++prelim;
		if (S[i] > maxConf) { maxConf = S[i]; maxIdx = i; }
	}
	std::cout << "[PARSE] anchors N=" << N << " C=" << C
			<< " prelim_pass=" << prelim
			<< " maxConf=" << maxConf << " @i=" << maxIdx << "\n";
	if (maxIdx >= 0 && C >= 4) {
		std::cout << "[PARSE] maxConf xywh=("
				<< X[maxIdx] << "," << Y[maxIdx] << ","
				<< W[maxIdx] << "," << H[maxIdx] << ")\n";
	}
#endif

	// Example decode for xywh in model-input pixels:
	for (int i = 0; i < N; ++i) {
		float conf = S[i];
		if (conf < confThresh) continue;

		float x = X ? X[i] : 0.f;
		float y = Y ? Y[i] : 0.f;
		float w = W ? W[i] : 0.f;
		float h = H ? H[i] : 0.f;

#if DEBUG_DNN
		if (dets.size() < 5) {
			std::cout << "[PARSE] raw[" << i << "] conf=" << conf
					<< " xywh=(" << x << "," << y << "," << w << "," << h << ")\n";
		}
#endif
		// convert xywh (center) -> corners in letterboxed space
		float x1 = x - w * 0.5f;
		float y1 = y - h * 0.5f;
		float x2 = x + w * 0.5f;
		float y2 = y + h * 0.5f;

		// undo letterbox back to original frame
		x1 = (x1 - dx) / scale;
		y1 = (y1 - dy) / scale;
		x2 = (x2 - dx) / scale;
		y2 = (y2 - dy) / scale;

		// clamp
		x1 = std::clamp(x1, 0.f, (float)imgW - 1);
		y1 = std::clamp(y1, 0.f, (float)imgH - 1);
		x2 = std::clamp(x2, 0.f, (float)imgW - 1);
		y2 = std::clamp(y2, 0.f, (float)imgH - 1);

#if DEBUG_DNN
		if (dets.size() < 5) {
			std::cout << "        map -> xyxy=(" << x1 << "," << y1 << "," << x2 << "," << y2 << ")\n";
		}
#endif
		Detection d;
		d.class_id = 0;      // single-class apple; adjust if you have more classes
		d.score    = conf;
		d.box      = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
		dets.push_back(d);
	}

#if DEBUG_DNN
	std::cout << "[PARSE] produced " << dets.size() << " raw boxes before NMS\n";
#endif

	// (your existing NMS here)
	// After NMS, optionally print the first few:
#if DEBUG_DNN
	// Suppose result vector is named 'finals'; else just use 'dets' if you do in-place NMS.
	const auto& finals = dets; // replace if different
	for (size_t k = 0; k < finals.size() && k < 5; ++k) {
		const auto& b = finals[k];
		std::cout << "[PARSE] keep[" << k << "] class=" << b.class_id
				<< " conf=" << b.score
				<< " rect=" << b.box << "\n";
	}
	std::cout << "[PARSE] exit\n";
#endif
	return dets;
}

// Returns the path to the output directory.
// For now this is hard coded to the directory the executable is executed in + /output.
static fs::path outputDirectory() {
	// Build the filepath string
	fs::path out = fs::current_path() / "output";
	error_code ec;
	// Check if the path exists
	if (!fs::exists(out, ec)) {
		// If the path did not exist, create the directory
		fs::create_directories(out, ec);
		if (ec) {
			cerr << "Failed to create output dir: " << out << " (" << ec.message() << " )\n";
		}
	}
	// Return the path
	return out;
}

// Generates a recording directory
static fs::path recordingDirectory() {
	// Build the filepath string
	fs::path out = outputDirectory() / ("recording_" + timestamp());
	error_code ec;
	// Check if the path exists
	if (!fs::exists(out, ec)) {
		// If the path did not exist create the directory
		fs::create_directories(out, ec);
		if (ec) {
			cerr << "Failed to create recording dir: " << out << " (" << ec.message() << " )\n";
		}
	}
	// Return the path
	return out;
}

// Saves the current frame to the output directory
static bool saveSnapshot(const Mat& frame, const fs::path& outdir, std::string* outPathStr = nullptr)
{
	if (frame.empty()) {
		cerr << "Snapshot failed: empty frame.\n";
		return false;
	}
	fs::path filepath = outdir / ("snapshot_" + timestamp() + ".png");
	if (imwrite(filepath.string(), frame)) {
		cout << "Saved snapshot: " << filepath.string() << "\n";
		if (outPathStr) *outPathStr = filepath.string();
		return true;
	} else {
		cerr << "imwrite failed for: " << filepath.string() << "\n";
		return false;
	}
}

// This will execute a command and capture and return its output.
string runCommand(const string& cmd) {

// If ran on windows
#ifdef _WIN32

	// Redirect STDERR into STDOUT for the command.
	string full = cmd + " 2>&1";

	FILE* pipe = _popen(full.c_str(), "r");

	if (!pipe) {
		cerr << "Unable to open command pipe.\n";
		return {};
	}

	string output;

	char buffer[4096];
	while (fgets(buffer, sizeof(buffer), pipe)) {
		output += buffer;
	}

	_pclose(pipe);
	return output;

#else

	cerr << "OS unsupported!\n"

#endif

}

vector<VideoDevice> fetchVideoDevices() {

	vector<VideoDevice> deviceList;

#ifdef _WIN32

	// FFmpeg command to list DirectShow devices
	const string cmd = "ffmpeg -hide_banner -f dshow -list_devices true -i dummy";

	const string result = runCommand(cmd);
	if (result.empty()) {
		cerr << "FFmpeg output empty\n";
		return deviceList;
	}

	regex quoted_name_re(R"raw("([^"]+)")raw");

	int deviceLineCounter = 0;

	// Parse the output
	size_t start = 0;
	while (start < result.size()) {

		size_t end = result.find('\n', start);

		if (end == string::npos) {
			end = result.size();
		}

		string line = result.substr(start, end - start);
		string lower = line;

		// Convert the line to lowercase
		for (char& c : lower) c = (char)tolower((unsigned char)c);

		//cout << "Line: " << lower << "\n";

		smatch m;
		if (regex_search(line, m, quoted_name_re)) {

			string deviceName = m[1].str();

			// If this is a video device
			if (lower.find("video") != string::npos) {

				// Add the device and its index to the vector
				deviceList.push_back(VideoDevice{ deviceLineCounter, deviceName});

				//cout << "Index: " << deviceLineCounter << " Name: " << deviceName << "\n";

				deviceLineCounter++;

			} else if (lower.find("audio") != string::npos) {

				// This is an audio device, we should increment the deviceLineCounter
				deviceLineCounter++;

			}

		}

		start = end + (end < result.size() ? 1 : 0);

	}

#else

	cerr << "OS unsupported!\n"

#endif

	return deviceList;

}

void printOptions(vector<VideoDevice> deviceList) {
	for (const VideoDevice& device : deviceList) {
		cout << "Index " << device.index << ": " << device.name << "\n";
	}
}

// Gonna be honest, I got this from Chat GPT
// The goal is to clear the terminal output
// But preserve the history.
void clearOutput() {

#ifdef _WIN32

	HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
	if (h == INVALID_HANDLE_VALUE) return;

	CONSOLE_SCREEN_BUFFER_INFO csbi{};
	if (!GetConsoleScreenBufferInfo(h, &csbi)) return;

	SHORT left   = csbi.srWindow.Left;
	SHORT top    = csbi.srWindow.Top;
	SHORT width  = csbi.srWindow.Right  - csbi.srWindow.Left + 1;
	SHORT height = csbi.srWindow.Bottom - csbi.srWindow.Top  + 1;

	DWORD written = 0;
	for (SHORT y = 0; y < height; ++y) {
		COORD pos = { left, static_cast<SHORT>(top + y) };
		FillConsoleOutputCharacterW(h, L' ', width, pos, &written);
		FillConsoleOutputAttribute(h, csbi.wAttributes, width, pos, &written);
	}
	SetConsoleCursorPosition(h, { left, top });

#else

	cerr << "OS unsupported!\n"

#endif

}

// Prompt the user to select a device
static int selectVideoDevice(vector<VideoDevice> deviceList) {

	// Validate that the list contains atleast one entry
	if (deviceList.size() <= 0) {
		cerr << "Invalid deviceList size!\n";
		return -1;
	}

	int index = -1;
	string input;

	int tries = 0;
	while ( tries < 3 ) {

		cout << "Select a Video Capture Device\n";

		printOptions(deviceList);

		cout << "Q - Quit\n";

		cout << "Enter device Index: ";

		cin >> input;

		string lower = input;

		for (char& c : lower) c = (char)tolower((unsigned char) c);

		if ( lower.find("q") != string::npos ) {
			// The user wishes to quit
			return -1;
		}

		try {
			index = stoi(input);
			return index;
		} catch ( const exception& ) {
			cerr << "Invalid index!\n";
			index = -1;
		}

		clearOutput();

		tries++;
	
	}

	return index;

}

static bool ocvBuiltWithCUDA() {
	try {
		// Fast path: CUDA device count > 0 implies CUDA runtime available AND OpenCV built with CUDA
		return getCudaEnabledDeviceCount() > 0;
	} catch (const Exception&) {
		// Fallback: parse build info text
		string info = getBuildInformation();

		// Regex: CUDA*YES
		regex cudaRegex(R"(CUDA\s*:\s*YES)", regex_constants::icase);

		return regex_search(info, cudaRegex);
	}
}

static bool ocvBuiltWithcuDNN() {
	string info = getBuildInformation();

	// Regex: CUDNN*YES
	regex cudnnRegex(R"(CUDNN\s*:\s*YES)", regex_constants::icase);

	return regex_search(info, cudnnRegex);
}

static void ocvPrintBuildInfo() {
	string info = getBuildInformation();
	cout << info << endl;

	return;
}

static void printCUDAReport() {
	cout << "OpenCV version: " << CV_VERSION << "\n";
	cout << "Built with CUDA?  " << (ocvBuiltWithCUDA()  ? "YES" : "NO") << "\n";
	cout << "Built with cuDNN? " << (ocvBuiltWithcuDNN() ? "YES" : "NO") << "\n";

	try {
		int n = getCudaEnabledDeviceCount();
		cout << "CUDA devices visible: " << n << "\n";
		if (n > 0) {
			DeviceInfo dev0(0);
			cout << "Device 0: " << dev0.name() << ", CC " << dev0.majorVersion() << "." << dev0.minorVersion() << "\n";
		}
	} catch (const Exception& e) {
		// Thrown if OpenCV isn’t built with CUDA or CUDA runtime not present
		cout << "CUDA query failed: " << e.what() << "\n";
	}

	if (ocvBuiltWithCUDA()) {
		printCudaDeviceInfo(0);
	}
}

// Callback function for trackbars to work on canny view.
void onTrackbarThresholds(int pos, void* userData) {
	auto* ctx = static_cast<CannyUIContext*>(userData);
	if (!ctx || !ctx->gray || !ctx->canny) return;
	if (ctx->gray->empty()) return;

	if (*(ctx->highThreshold) < *(ctx->lowThreshold)) *(ctx->highThreshold) = *(ctx->lowThreshold);

	Canny(*(ctx->gray), *(ctx->canny), *(ctx->lowThreshold), *(ctx->highThreshold), *(ctx->kernel));
	imshow(ctx->window, *(ctx->canny));
}

// Kernel slider moved: remap and adjust threshold maxes
void onTrackbarKernel(int pos, void* userdata) {
	auto* ctx = static_cast<CannyUIContext*>(userdata);
	if (!ctx) return;

	// Decide new max based on kernel size.
	// (Heuristic: scale with ksize; adjust to your taste.)
	// 3 → 255, 5 → 425, 7 → 595
	int newMax;

	switch (*(ctx->kernelIndex)) {
		case 0:
			newMax = 255;
			*(ctx->kernel) = 3;
			break;
		case 1:
			newMax = 4095;
			*(ctx->kernel) = 5;
			break;
		case 2:
			newMax = 35565;
			*(ctx->kernel) = 7;
			break;
		default:
			cerr << "Error with kernel size!\n";
			return;
	}

	// Update trackbar ranges dynamically
	setTrackbarMax(TB_LOW,  ctx->window, newMax);
	setTrackbarMax(TB_HIGH, ctx->window, newMax);

	// Clamp current values to the new max
	if (*(ctx->lowThreshold)  > newMax) *(ctx->lowThreshold)  = newMax;
	if (*(ctx->highThreshold) > newMax) *(ctx->highThreshold) = newMax;

	setTrackbarPos(TB_LOW,  ctx->window, *(ctx->lowThreshold));
	setTrackbarPos(TB_HIGH, ctx->window, *(ctx->highThreshold));

	// Recompute with updated settings
	onTrackbarThresholds(0, userdata);
}

int loadModel(string path, dnn::Net* net, bool diagnostics) {

	if (diagnostics) {
		// 3.1) Turn on very chatty OpenCV logs (especially helpful for dnn backtrace)
		cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);

		// 3.2) Optional: disable some dnn memory opts to get clearer shapes in logs
		_putenv("OPENCV_DNN_DISABLE_MEMORY_OPTIMIZATIONS=1");
		_putenv("OPENCV_LOG_LEVEL=VERBOSE");

		// 3.3) Report model file size
		const std::string onnxPath = "best.onnx";  // <-- keep your existing relative/absolute path here
		try {
			if (fs::exists(onnxPath)) {
				auto sz = fs::file_size(onnxPath);
				std::cout << "[DNN] best.onnx size: " << humanBytes(sz) << "\n";
			} else {
				std::cout << "[DNN] ERROR: File not found: " << onnxPath << "\n";
			}
		} catch (const std::exception& e) {
			std::cout << "[DNN] Could not stat file size: " << e.what() << "\n";
		}
	}

	// 3.4) Load the model
	try {
		*net = dnn::readNetFromONNX(path);
		std::cout << "[DNN] Loaded ONNX model.\n";
	} catch (const cv::Exception& e) {
		std::cerr << "[DNN] Failed to readNetFromONNX: " << e.err << " | " << e.func << " | " << e.msg << "\n";
		throw; // bail early — nothing else to do
	}

	if (diagnostics) {
		// 3.5) Print a compact summary of the network as OpenCV sees it
		printNetSummary(*net);

		// 3.6) Probe shapes (best-effort; safe if it throws)
		probeLayerShapes(*net);
	}
}

int main() {

	printCUDAReport();

	ocvPrintBuildInfo();

	vector<VideoDevice> deviceList = fetchVideoDevices();

	int index = selectVideoDevice(deviceList);

	if (index < 0) {
		cerr << "Invalid device Index " << index << "\n";
		return -1;
	}

	VideoCapture cap(index, CAP_DSHOW);

	if (!cap.isOpened()) {
		cerr << "Failed to open device\n";
		return -1;
	}

	Recorder rec;

	cap.set(CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
	cap.set(CAP_PROP_FPS, 60);

	bool quit = false;

	Mat frame;
	Mat gray_frame;
	Mat canny_frame;

	int lowThreshold = 25;
	int highThreshold = 25;
	int kernelIndex = 0;
	int kernelSize = 3;
	int mode = 0;
	int prevMode = -1;
	fs::path outDir = outputDirectory();
	fs::path recDir;
	int recFrame = 0;
	int recRate = 8;

	const char* WIN_NAME = "Display";
	namedWindow(WIN_NAME, WINDOW_AUTOSIZE);

	bool cannyUIReady = false;
	CannyUIContext ctx { &gray_frame, &canny_frame, &lowThreshold, &highThreshold, &kernelIndex, &kernelSize, WIN_NAME};

	bool modelReady = false;
	dnn::Net net;

	
	cout << "Model Path: " << model.string() << "\n";

	for(;;) {

		if(!cap.read(frame) || frame.empty()) break;

		// Mode has changed
		if (mode != prevMode) {
			if (mode == CANNY && !cannyUIReady) {
				// Create trackbars with initial max; these will be updated by kernel callback
				createTrackbar(TB_LOW,  WIN_NAME, &lowThreshold, 255, onTrackbarThresholds, &ctx);
				createTrackbar(TB_HIGH, WIN_NAME, &highThreshold, 255, onTrackbarThresholds, &ctx);

				// Kernel slider (0..2). use dedicated callback.
				createTrackbar(TB_KER, WIN_NAME, &kernelIndex, 2, onTrackbarKernel, &ctx);

				// Initialize maxes based on current kernel and render once
				onTrackbarKernel(kernelIndex, &ctx);
				cannyUIReady = true;
			} else if ( mode == APPLES && !modelReady ) {

				int ret = loadModel(model.string(), &net, true);

				// If built with cuda/cudnn
				if ( ocvBuiltWithCUDA() && ocvBuiltWithcuDNN() ) {
					cout << "Using DNN GPU\n";
					net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
					net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
				} else {
					cout << "Using DNN CPU\n";
					net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
					net.setPreferableTarget(dnn::DNN_TARGET_CPU);
				}

				modelReady = true;

			}
			prevMode = mode;
		}

		// Display switch
		// Displays regular frame or canny depending on mode value.
		switch (mode) {
			case DEFAULT: // default
				imshow(WIN_NAME, frame);
				break;
			case CANNY: // canny
				// Convert the frame to canny
				cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
				blur(gray_frame, canny_frame, Size(3, 3));
				Canny(canny_frame, canny_frame, lowThreshold, highThreshold, kernelSize);

				// Generate and display the canny view
				onTrackbarThresholds(0, &ctx);
				break;
			case APPLES: // Apple detection
				/*
				float scale; int dx, dy;
				cv::Mat inp = letterbox(frame, modelResolution, scale, dx, dy);
				cv::Mat blob = cv::dnn::blobFromImage(inp, 1.0/255.0, cv::Size(modelResolution, modelResolution), cv::Scalar(), true, false);
				*/
				cv::Mat out;

				// Reshape the frame so that it will fit in the model.
				// Model expects 1920x1920
				constexpr int MODEL_W = 1920;
				constexpr int MODEL_H = 1920;

				float scale = 1.f; int dx = 0, dy = 0;
				cv::Mat input = letterbox_reshape(frame, MODEL_W, MODEL_H, scale, dx, dy);

				cv::Mat blob = cv::dnn::blobFromImage(
					input, 1.0/255.0, cv::Size(MODEL_W, MODEL_H),
					cv::Scalar(), /*swapRB=*/true, /*crop=*/false
				);

				printBlobInfo(blob, "[DNN] Input");

				cout << "[DNN] setInput(...)\n";
				net.setInput(blob);

				try {
					cout << "[DNN] forward() starting...\n";
					TickMeter tm; tm.start();
					out = net.forward();
					tm.stop();
					cout << "[DNN] forward() OK in " << tm.getTimeMilli() << " ms\n";

					// Analyze the output from net.forward()
					debugAnalyzeOut(out, /*modelW*/1920, /*modelW*/1920, confidence);

					// Log output blob shapes
					printBlobInfo(out, "[DNN] Output");
				} catch (const Exception& e) {
					std::cerr << "\n[DNN] forward() CRASH/ERROR\n";
					std::cerr << "  err : " << e.err  << "\n";
					std::cerr << "  func: " << e.func << "\n";
					std::cerr << "  msg : " << e.msg  << "\n";

					// Dump summary again (sometimes layer state gets clearer after setInput)
					std::cerr << "\n[DNN] Net summary after setInput (for context):\n";
					printNetSummary(net);

					std::cerr << "\n[DNN] If msg mentions shape mismatch, check the first conv/input layer above.\n";
					throw; // rethrow so your outer error handling can catch/log it as well
				}
				//cv::Mat out = net.forward();
				auto dets = parseDetections(out, confidence, intersection, modelResolution, frame.cols, frame.rows, scale, dx, dy);

				for (const auto& d : dets) {
					cv::rectangle(frame, d.box, {0,255,0}, 2);
					char buf[64]; std::snprintf(buf, sizeof(buf), "id=%d %.2f", d.class_id, d.score);
					cv::putText(frame, buf, d.box.tl() + cv::Point(0,-5), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
				}

				imshow(WIN_NAME, frame);
				break;
		}

		// If recording is active
		if (rec.active) {

			// Write the current modes frame to the file.
			// Consider rewriting the display logic to use a shared mat
			// Doing this would remove the need for these switch cases.
			switch (mode) {
				case DEFAULT: // default
				case APPLES: // Apple Detection
					// Write the current frame to the file.
					rec.write(frame);
					break;
				case CANNY: // canny
					// Write the current canny_frame to the file.
					rec.write(canny_frame);
					break;
			}

			// If save_frames is true periodicaly save a screenshot
			if ( (recFrame++ % recRate == 0) && rec.save_frames ) {

				// Save the current modes frame
				// Consider rewriting the display logic to use a shared mat
				// Doing this would remove the need for these switch cases.
				switch (mode) {
				case DEFAULT: // default
				case APPLES: // Apple Detection
					// Save the screenshot
					saveSnapshot(frame, recDir, nullptr);
					break;
				case CANNY: // canny
					// Save the screenshot
					saveSnapshot(canny_frame, recDir, nullptr);
					break;
				}

			}

		}

		// This routine will process any keyboard inputs.
		int key = waitKey(1);
		switch (key) {
			case 27: // "ESC" to quit
				quit = true;
				break;
			case 'M': // Cycle Modes
			case 'm': // Cycle Modes
				prevMode = mode++;
				//mode++;
				if ( mode >= MODES_END ) {
					mode = 0;
				}
				break;
			case 'p': // Screenshot
			case 'P': // Screenshot
				switch (mode) {
				case DEFAULT: // default
				case APPLES: // Apple Detection
					// Save the screenshot
					saveSnapshot(frame, recDir, nullptr);
					break;
				case CANNY: // canny
					// Save the screenshot
					saveSnapshot(canny_frame, recDir, nullptr);
					break;
				}
				break;
			case 'R': // Recording while Saving frames periodicaly
				rec.save_frames = true;
			case 'r': // Regular recording
				if (rec.active) {
					rec.stop();
				} else {
					int width  = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
					int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
					double fps = cap.get(CAP_PROP_FPS);
					if (fps <= 0 || fps > 240) fps = 30.0; // sane default
					recDir = recordingDirectory();
					if (!rec.start(width, height, fps, recDir, mode)) {
						cerr << "Recording could not be started.\n";
					}
				}

		}
		if ( quit ) break;
	}

	return 0;
}