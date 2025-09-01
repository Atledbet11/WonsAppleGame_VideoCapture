#ifdef _WIN32
#include <windows.h>
#endif

#include "video_display.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

static inline double now_ms_() {
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// ---------- Utilities ----------
std::string video_display::exe_dir_() {
#ifdef _WIN32
	wchar_t buf[MAX_PATH];
	DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
	std::wstring wpath(buf, len);
	std::filesystem::path p(wpath);
#else
	// Best-effort; many Linux distros expose /proc/self/exe
	std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe");
#endif
	return p.parent_path().string();
}

std::string video_display::timestamp_ms_() {
	using namespace std::chrono;
	const auto now = system_clock::now();
	const auto t = system_clock::to_time_t(now);
	std::tm tm{};
#ifdef _WIN32
	localtime_s(&tm, &t);
#else
	localtime_r(&t, &tm);
#endif
	auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
	std::ostringstream oss;
	oss << std::put_time(&tm, "%Y%m%d_%H%M%S")
		<< "_" << std::setw(3) << std::setfill('0') << ms.count();
	return oss.str();
}

bool video_display::ensure_dir_(const std::string& path) {
	std::error_code ec;
	if (std::filesystem::exists(path)) return true;
	return std::filesystem::create_directories(path, ec) || std::filesystem::exists(path);
}

// ---------- Ctor / Dtor ----------
video_display::video_display() {
	output_dir = exe_dir_() + std::string("/output");
	ensure_dir_(output_dir);
}

video_display::~video_display() {
	stop();
	stopRecording();
	// Stop screenshot thread last
	if (screenshot_running_.load()) {
		screenshot_running_.store(false);
		screenshot_cv_.notify_all();
		if (screenshot_th_.joinable()) screenshot_th_.join();
	}
}

// ---------- Public API ----------
bool video_display::start(const std::string& window_name) {
	if (running_.load()) return true;
	window_name_ = window_name;

	// Start screenshot worker (lazy-start if not already)
	if (!screenshot_running_.load()) {
		screenshot_running_.store(true);
		screenshot_th_ = std::thread(&video_display::screenshot_worker_, this);
	}

	running_.store(true);
	display_th_ = std::thread(&video_display::display_loop_, this, window_name_);
	return true;
}

void video_display::stop() {
	if (!running_.load()) return;
	running_.store(false);
	if (display_th_.joinable()) display_th_.join();
	if (!window_name_.empty()) {
		try { destroyWindow(window_name_); } catch (...) {}
	}
}

void video_display::set_input_callback(std::function<void(int)> cb) {
	std::lock_guard<std::mutex> lk(cb_mtx_);
	on_key_ = std::move(cb);
}

bool video_display::startRecording(int fps_hint, int fourcc, int width, int height) {
	if (recording_.load()) return true;

	rec_fps_hint_ = fps_hint;
	rec_fourcc_   = fourcc;
	rec_w_        = width;
	rec_h_        = height;

	// Create the writer on the worker thread (to bind to its lifetime)
	recording_.store(true);
	recording_th_ = std::thread(&video_display::recording_worker_, this);
	return true;
}

void video_display::stopRecording() {
	if (!recording_.load()) return;
	recording_.store(false);
	recording_cv_.notify_all();
	if (recording_th_.joinable()) recording_th_.join();
	// Close writer just in case
	if (writer_.isOpened()) writer_.release();
}

void video_display::saveScreenshot() {
	cv::Mat snap;
	{
		std::lock_guard<std::mutex> lk(frame_mtx_);
		if (frame.empty()) return;
		snap = frame.clone();
	}
	ensure_dir_(output_dir);          // ensure at call-time too
	enqueue_screenshot_(snap);
}

bool video_display::cloneFrame(cv::Mat& dst) {
	std::lock_guard<std::mutex> lk(frame_mtx_);
	if (frame.empty()) return false;
	dst = frame.clone();
	return true;
}

void video_display::setFrame(const cv::Mat& bgr) {
	std::lock_guard<std::mutex> lk(frame_mtx_);
	frame = bgr.clone();
}

void video_display::display_loop_(std::string window_name) {
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

	using clock = std::chrono::steady_clock;
	auto  t0 = clock::now();
	int   frames = 0;

	while (running_.load()) {
		cv::Mat to_show; std::string diag; std::string diag2;
		{
			std::lock_guard<std::mutex> lk(frame_mtx_);
			if (!frame.empty()) to_show = frame.clone();
			diag = diagnostic_line1;
			diag2 = diagnostic_line2;
		}

		if (!to_show.empty()) {
			// --- FPS measurement (update ~2x/sec), draw every frame ---
			frames++;
			auto   now = clock::now();
			auto   ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
			if (ms >= 500) {
				last_fps_ = (ms > 0) ? (frames * 1000.0 / static_cast<double>(ms)) : 0.0;
				frames = 0;
				t0 = now;
			}

			if (draw_fps_.load()) {
				cv::putText(to_show, cv::format("FPS: %.1f", std::max(0.0, last_fps_)),
							{12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255,255,255}, 2, cv::LINE_AA);
			}

			if (draw_diag_.load() && !diag.empty()) {
				cv::putText(to_show, diag, {12, 54},
							cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2, cv::LINE_AA);
			}

			if (draw_diag_2_.load() && !diag2.empty()) {
				cv::putText(to_show, diag2, {12, 80},
							cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2, cv::LINE_AA);
			}

			if (recording_.load()) {
				enqueue_record_frame_(to_show);
			}

			cv::imshow(window_name, to_show);
		}

		int key = cv::waitKey(1);
		if (key >= 0) {
			key &= 0xFF; // normalize
			std::function<void(int)> cb_copy;
			{ std::lock_guard<std::mutex> lk(cb_mtx_); cb_copy = on_key_; }
			if (cb_copy) cb_copy(key);
		}
	}
}

// ---------- Internal: screenshot worker ----------
void video_display::enqueue_screenshot_(const cv::Mat& bgr) {
	{
		std::lock_guard<std::mutex> lk(screenshot_mtx_);
		screenshot_q_.push(bgr.clone());
	}
	screenshot_cv_.notify_one();
}

void video_display::screenshot_worker_() {
	while (screenshot_running_.load()) {
		std::unique_lock<std::mutex> lk(screenshot_mtx_);
		screenshot_cv_.wait(lk, [&]{ return !screenshot_running_.load() || !screenshot_q_.empty(); });
		if (!screenshot_running_.load()) break;
		auto img = screenshot_q_.front();
		screenshot_q_.pop();
		lk.unlock();

		if (!img.empty()) {
			const std::string name = output_dir + "/screenshot_" + timestamp_ms_() + ".png";
			try {
				imwrite(name, img);
			} catch (...) {
				// swallow errors to keep thread alive
			}
		}
	}
}

// ---------- Internal: recording worker ----------
void video_display::enqueue_record_frame_(const cv::Mat& bgr) {
	{
		std::lock_guard<std::mutex> lk(recording_mtx_);
		// Prevent unbounded growth; drop frames if queue gets too large
		if (recording_q_.size() > 120) { // ~2 seconds at 60fps
			// drop oldest
			recording_q_.pop();
		}
		recording_q_.push(bgr.clone());
	}
	recording_cv_.notify_one();
}

void video_display::recording_worker_() {
	// Open writer lazily on first frame (so we know W/H)
	while (true) {
		std::unique_lock<std::mutex> lk(recording_mtx_);
		recording_cv_.wait(lk, [&]{
			return !recording_.load() || !recording_q_.empty();
		});

		if (!recording_.load() && recording_q_.empty()) break;

		cv::Mat img = std::move(recording_q_.front());
		recording_q_.pop();
		lk.unlock();

		if (img.empty()) continue;

		if (!writer_.isOpened()) {
			// Ensure output dir each time in case user changed it at runtime
			ensure_dir_(output_dir);

			int W = rec_w_ > 0 ? rec_w_ : img.cols;
			int H = rec_h_ > 0 ? rec_h_ : img.rows;

			const std::string base = output_dir + "/record_" + timestamp_ms_();
			std::string path = base + ".mp4";

			// Try MP4 (mp4v) first
			bool opened = writer_.open(path, rec_fourcc_, rec_fps_hint_, cv::Size(W, H), true);
			if (!opened) {
				// Fallback to AVI with MJPG
				path = base + ".avi";
				opened = writer_.open(path, cv::VideoWriter::fourcc('M','J','P','G'),
									rec_fps_hint_, cv::Size(W, H), true);
			}

			if (!opened) {
				std::cerr << "[recording] Failed to open writer (mp4+avi). Stopping recording.\n";
				std::lock_guard<std::mutex> lk2(recording_mtx_);
				std::queue<cv::Mat> empty; std::swap(recording_q_, empty);
				recording_.store(false);
				continue;
			} else {
				std::cerr << "[recording] Writing to: " << path << "\n";
			}
		}

		try {
			writer_.write(img);
		} catch (const std::exception& e) {
			std::cerr << "[recording] write failed: " << e.what() << "\n";
			if (writer_.isOpened()) writer_.release();
			std::lock_guard<std::mutex> lk2(recording_mtx_);
			std::queue<cv::Mat> empty; std::swap(recording_q_, empty);
			recording_.store(false);
		} catch (...) {
			std::cerr << "[recording] write failed: unknown error\n";
			if (writer_.isOpened()) writer_.release();
			std::lock_guard<std::mutex> lk2(recording_mtx_);
			std::queue<cv::Mat> empty; std::swap(recording_q_, empty);
			recording_.store(false);
		}
	}

	if (writer_.isOpened()) writer_.release();
}
