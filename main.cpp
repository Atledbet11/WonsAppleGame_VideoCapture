
/*
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
//#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;
//using namespace dnn;
//using namespace cuda;

int main(int, char**) {

    cout << "Startup\n";

    cout << cv::getBuildInformation() << endl;

    //printCudaDeviceInfo(0);

    cout << "Hello, world!\n";
    
}

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
  try {
    std::cout << "OpenCV: " << CV_VERSION << "\n";
    std::cout << cv::getBuildInformation() << std::endl; // shows "Use of CUDA: YES/NO"
    int n = 0;
    try { n = cv::cuda::getCudaEnabledDeviceCount(); } catch (...) {}
    std::cout << "CUDA devices: " << n << "\n";

    // If you're using video files/camera, test minimally:
    cv::VideoCapture cap(0, cv::CAP_MSMF); // or CAP_DSHOW
    if(!cap.isOpened()) std::cerr << "Camera open failed\n";

    return 0;
  } catch (const cv::Exception& e) {
    std::cerr << "OpenCV EXCEPTION:\n" << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "STD EXCEPTION:\n" << e.what() << std::endl;
    return 1;
  }
}

*/

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <regex>

static bool ocvBuiltWithCUDA() {
    try {
        // Fast path: CUDA device count > 0 implies CUDA runtime available AND OpenCV built with CUDA
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    } catch (const cv::Exception&) {
        // Fallback: parse build info text
        std::string info = cv::getBuildInformation();

        // Regex: CUDA*YES
        std::regex cudaRegex(R"(CUDA\s*:\s*YES)", std::regex_constants::icase);

        return std::regex_search(info, cudaRegex);
    }
}

static bool ocvBuiltWithcuDNN() {
    std::string info = cv::getBuildInformation();

    // Regex: CUDNN*YES
    std::regex cudnnRegex(R"(CUDNN\s*:\s*YES)", std::regex_constants::icase);

    return std::regex_search(info, cudnnRegex);
}

static void ocvPrintBuildInfo() {
  std::string info = cv::getBuildInformation();
  std::cout << info << std::endl;

  return;
}

static void printCUDAReport() {
    std::cout << "OpenCV version: " << CV_VERSION << "\n";
    std::cout << "Built with CUDA?  " << (ocvBuiltWithCUDA()  ? "YES" : "NO") << "\n";
    std::cout << "Built with cuDNN? " << (ocvBuiltWithcuDNN() ? "YES" : "NO") << "\n";

    try {
        int n = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "CUDA devices visible: " << n << "\n";
        if (n > 0) {
            cv::cuda::DeviceInfo dev0(0);
            std::cout << "Device 0: " << dev0.name() << ", CC " << dev0.majorVersion() << "." << dev0.minorVersion() << "\n";
        }
    } catch (const cv::Exception& e) {
        // Thrown if OpenCV isn’t built with CUDA or CUDA runtime not present
        std::cout << "CUDA query failed: " << e.what() << "\n";
    }

    if (ocvBuiltWithCUDA()) {
      cv::cuda::printCudaDeviceInfo(0);
    }
}

int main() {

  printCUDAReport();

  return 0;
}