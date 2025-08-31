#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <string>

class video_display {
public:
	// ---- Public data (as requested) ----
	cv::Mat frame;                 // latest frame to display (you may write to it)
	std::string diagnostic;        // text to draw under the FPS label
	std::string output_dir;        // public output directory; defaults to <exe>/output

	// ---- Lifecycle ----
	video_display();
	~video_display();

	// Start/stop the display thread (creates a window named window_name)
	bool start(const std::string& window_name = "Preview");
	void stop();

	// ---- User input (key) callback ----
	// Callback receives the key code returned by cv::waitKey (or -1 if none).
	void set_input_callback(std::function<void(int)> cb);

	// ---- Recording controls (independent writer thread) ----
	// If width/height are zero, the first frame size seen is used.
	bool startRecording(int fps_hint = 60, int fourcc = cv::VideoWriter::fourcc('m','p','4','v'),
						int width = 0, int height = 0);
	void stopRecording();
	bool isRecording() const noexcept { return recording_.load(); }

	// ---- Screenshot (queued to saver thread) ----
	void saveScreenshot();

	// ---- Optional overlays ----
	void set_draw_fps(bool enabled) noexcept { draw_fps_.store(enabled); }
	void set_draw_diag(bool enabled) noexcept { draw_diag_.store(enabled); }

	// Safe clones (helpful if you prefer not to touch the public members directly)
	bool cloneFrame(cv::Mat& dst);
	void setFrame(const cv::Mat& bgr);

private:
	// Display
	void display_loop_(std::string window_name);

	// Screenshot saver
	void screenshot_worker_();
	void enqueue_screenshot_(const cv::Mat& bgr);

	// Recording writer
	void recording_worker_();
	void enqueue_record_frame_(const cv::Mat& bgr);

	// Utilities
	static std::string exe_dir_();
	static std::string timestamp_ms_();    // YYYYmmdd_HHMMSS_mmm
	static bool ensure_dir_(const std::string& path);

	// Threads & flags
	std::thread display_th_;
	std::atomic<bool> running_{false};
	std::string window_name_;

	// Input callback
	std::function<void(int)> on_key_;
	std::mutex cb_mtx_;

	// Overlays
	double last_fps_ = 0.0;
	std::atomic<bool> draw_fps_{true};
	std::atomic<bool> draw_diag_{true};

	// Frame access guard (protects reads/writes to public frame/diagnostic during display copy)
	std::mutex frame_mtx_;

	// ---- Screenshot queue/thread ----
	std::thread screenshot_th_;
	std::atomic<bool> screenshot_running_{false};
	std::queue<cv::Mat> screenshot_q_;
	std::mutex screenshot_mtx_;
	std::condition_variable screenshot_cv_;

	// ---- Recording queue/thread ----
	std::thread recording_th_;
	std::atomic<bool> recording_{false};
	std::queue<cv::Mat> recording_q_;
	std::mutex recording_mtx_;
	std::condition_variable recording_cv_;
	cv::VideoWriter writer_;
	int rec_w_ = 0, rec_h_ = 0, rec_fourcc_ = 0, rec_fps_hint_ = 60;
};