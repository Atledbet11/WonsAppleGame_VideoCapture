#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <regex>
#include <vector>
#include <filesystem>
#include <algorithm> // For std::max

#define NOMINMAX
#include <windows.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

namespace fs = std::filesystem;

// Trackbar names (keep consistent)
static const char* TB_LOW   = "Low Threshold";
static const char* TB_HIGH  = "High Threshold";
static const char* TB_KER   = "Kernel(3/5/7)";

enum modes {
	DEFAULT,
	CANNY,
	MODES_END
};

struct VideoDevice {
	int index;
	string name;
};

struct CannyUIContext {
	Mat* gray;				// Gray Frame
	Mat* canny;				// Canny Frame
	int* lowThreshold;
	int* highThreshold;
	int* kernelIndex;
	int* kernel;
	const char* window;		// Window Name
};


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

struct Recorder {
	VideoWriter writer;
	bool active = false;
	bool save_frames = false;
	fs::path out_path;

	// Try MP4 first; fall back to AVI if needed
	bool start(int width, int height, double fps, const fs::path& outdir, int mode)
	{
		if (active) return true;

		bool isColor = true;

		switch (mode) {
			case CANNY:
				isColor = false;
				break;
		}

		int fourcc_mp4v = VideoWriter::fourcc('m','p','4','v');
		int fourcc_xvid = VideoWriter::fourcc('X','V','I','D');

		out_path = outdir / ("recording_" + timestamp() + ".mp4");
		bool ok = writer.open(out_path.string(), fourcc_mp4v, (fps > 0 ? fps : 30.0),
								Size(width, height), isColor);

		if (!ok) {
			// Fallback to AVI/XVID
			out_path = outdir / ("recording_" + timestamp() + ".avi");
			ok = writer.open(out_path.string(), fourcc_xvid, (fps > 0 ? fps : 30.0),
								Size(width, height), isColor);
		}

		if (!ok) {
			cerr << "Failed to open VideoWriter for recording.\n";
			return false;
		}

		active = true;
		cout << "Recording started: " << out_path.string() << "\n";
		return true;
	}

	void write(const Mat& frame)
	{
		if (active && !frame.empty()) {
			writer.write(frame);
		}
	}

	void stop()
	{
		if (active) {
			writer.release();
			active = false;
			save_frames = false;
			cout << "Recording stopped: " << out_path.string() << "\n";
		}
	}
};

static fs::path outputDirectory() {
	fs::path out = fs::current_path() / "output";
	error_code ec;
	if (!fs::exists(out, ec)) {
		fs::create_directories(out, ec);
		if (ec) {
			cerr << "Failed to create output dir: " << out << " (" << ec.message() << " )\n";
		}
	}
	return out;
}

static fs::path recordingDirectory() {
	fs::path out = outputDirectory() / ("recording_" + timestamp());
	error_code ec;
	if (!fs::exists(out, ec)) {
		fs::create_directories(out, ec);
		if (ec) {
			cerr << "Failed to create recording dir: " << out << " (" << ec.message() << " )\n";
		}
	}
	return out;
}

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

	//cout << "Provided index: " << index << "\n";

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

	for(;;) {

		if(!cap.read(frame)) break;

		// Mode has changed
		if (mode != prevMode) {
			if (mode == 1 && !cannyUIReady) {
				// Create trackbars with initial max; these will be updated by kernel callback
				createTrackbar(TB_LOW,  WIN_NAME, &lowThreshold, 255, onTrackbarThresholds, &ctx);
				createTrackbar(TB_HIGH, WIN_NAME, &highThreshold, 255, onTrackbarThresholds, &ctx);

				// Kernel slider (0..2). use dedicated callback.
				createTrackbar(TB_KER, WIN_NAME, &kernelIndex, 2, onTrackbarKernel, &ctx);

				// Initialize maxes based on current kernel and render once
				onTrackbarKernel(kernelIndex, &ctx);
				cannyUIReady = true;
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
		}

		// If recording is active
		if (rec.active) {

			// Write the current modes frame to the file.
			// Consider rewriting the display logic to use a shared mat
			// Doing this would remove the need for these switch cases.
			switch (mode) {
				case DEFAULT: // default
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