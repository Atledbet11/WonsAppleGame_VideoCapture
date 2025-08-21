
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
*/

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