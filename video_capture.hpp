#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/cuda.hpp> // Safe to include; runtime calls are guarded
#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

// forward-declare our utils
namespace utils { std::string runCommand(const std::string& cmd); }

class video_capture {
public:
	enum class Backend { CPU = 0, CUDA = 1 };
	enum class CapBackend { Auto, MSMF, DSHOW };

	struct StreamInfo {
		int    width          = 0;      // actual width after negotiation
		int    height         = 0;      // actual height after negotiation
		double fps_reported   = 0.0;    // CAP_PROP_FPS (often 0 on DirectShow until warmed)
		double fps_measured   = 0.0;    // measured by timing reads
		int    frames_counted = 0;      // frames used to estimate fps_measured
	};

	// ---- Public outputs (as requested) ----
	cv::Mat            out_cpu;     // last captured frame (CPU)
	cv::cuda::GpuMat   out_gpu;     // last uploaded frame (CUDA path)

	// ---- Config & state ----
	Backend   backend() const noexcept { return backend_; }
	void      set_backend(Backend b) noexcept { backend_ = b; }
	int       camera_index() const noexcept { return cam_index_; }
	void      set_camera_index(int idx) noexcept { cam_index_ = idx; }
	CapBackend opened_backend() const noexcept { return opened_backend_; }

	// Desired settings (used by negotiate)
	void set_desired(int w, int h, double fps) noexcept {
		desired_width_ = w; desired_height_ = h; desired_fps_ = fps;
	}
	void set_writer_format(int fourcc, const std::string& container_hint = {}) {
		writer_fourcc_ = fourcc; container_hint_ = container_hint;
	}

	// Negotiate with device (apply desired W/H/FPS, optionally force MJPG), then probe actuals.
	// Returns the measured/actual stream info.
	StreamInfo negotiate(bool prefer_mjpg = true, int warmup_frames = 5, int measure_ms = 1500);

	// Auto-negotiate toward a target, trying MSMF then DSHOW, MJPG, and common modes.
	// Returns the selected/achieved stream info.
	StreamInfo negotiate_auto(int target_w, int target_h, double min_fps,
							bool prefer_mjpg = true, int measure_ms = 1200);

	// Start/stop capture thread. Returns false if opening the device fails.
	bool start_capture();
	void stop_capture();

	// Thread-safe snapshots instead of using live frames
	// Thread-safe snapshots of the latest frame
	bool snapshot_cpu(cv::Mat& dst);                    // deep copy of out_cpu
	bool snapshot_gpu_download(cv::Mat& dst_cpu);       // lock + download out_gpu to CPU
	// (optional) device-to-device copy if you later want a GpuMat snapshot:
	bool snapshot_gpu(cv::cuda::GpuMat& dst_gpu);

	// Access the last-known actual stream info (after negotiate or after the first probe)
	StreamInfo stream_info() const noexcept { return info_; }

	// Prompt the user to pick a video device
	int selectVideoDevice();

	// Basic ctor/dtor
	video_capture()  = default;
	~video_capture() { stop_capture(); }

private:
	// Internal helpers
	bool     open_device_();
	bool     open_device_preferring_(CapBackend pref);
	bool     try_mode_(int w, int h, double fps, bool prefer_mjpg,
					int warmup_frames, int measure_ms, StreamInfo& out);
	void     capture_loop_();
	StreamInfo probe_stream_(int warmup_frames, int measure_ms);

	// Video device struct
	struct VideoDevice {
		int indexInList = -1;
		std::string name;
	};

	// Fetch device list
	std::vector<VideoDevice> fetchVideoDevices_();
	void printOptions_(const std::vector<VideoDevice>& list);
	void clearOutput_();

	// ---- Device & backend ----
	int       cam_index_       = 0;
	Backend   backend_         = Backend::CPU;
	CapBackend opened_backend_ = CapBackend::Auto;
	cv::VideoCapture cap_;

	// ---- Desired capture props ----
	int     desired_width_     = 0;      // 0 = don't force
	int     desired_height_    = 0;
	double  desired_fps_       = 0.0;

	// ---- Writer formats ----
	int         writer_fourcc_   = cv::VideoWriter::fourcc('M','J','P','G');
	std::string container_hint_; // e.g., "mp4", "avi" (not used here yet; stored for later)

	// ---- Threading ----
	std::thread         th_;
	std::atomic<bool>   running_{false};
	std::mutex          out_mutex_;  // protects out_cpu/out_gpu during write

	// ---- Last-known stream info ----
	StreamInfo          info_;
};