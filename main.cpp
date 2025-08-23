#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <regex>

#include <windows.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

struct VideoDevice {
	int index;
	string name;
};

// This will execute a command and capture and return its output.
string run_command(const string& cmd) {

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

	const string result = run_command(cmd);
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
		return 1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
	cap.set(CAP_PROP_FPS, 60);

	Mat frame;

	for(;;) {
		if(!cap.read(frame)) break;
		imshow("Capture", frame);
		if(waitKey(1) == 27) break;
	}

	return 0;
}