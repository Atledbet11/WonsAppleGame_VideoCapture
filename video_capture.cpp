#include "video_capture.hpp"
#include "utils.hpp"
#include <chrono>
#include <iostream>
#include <regex>
#include <algorithm>

using namespace std;

static inline double now_ms_() {
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

bool video_capture::open_device_() {
#ifdef _WIN32
	if (!cap_.open(cam_index_, cv::CAP_DSHOW))
		return false;
#else
	if (!cap_.open(cam_index_))
		return false;
#endif
	return cap_.isOpened();
}

video_capture::StreamInfo video_capture::probe_stream_(int warmup_frames, int measure_ms) {
	StreamInfo si{};

	// Warm-up
	cv::Mat frame;
	for (int i = 0; i < warmup_frames; ++i) {
		if (!cap_.read(frame)) break;
	}
	if (frame.empty()) {
		cap_.read(frame); // try once more
	}

	if (!frame.empty()) {
		si.width  = frame.cols;
		si.height = frame.rows;
	} else {
		si.width  = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
		si.height = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
	}

	si.fps_reported = cap_.get(cv::CAP_PROP_FPS);

	// Measure
	int frames = 0;
	const double t0 = now_ms_();
	while (true) {
		cv::Mat f;
		if (!cap_.read(f)) break;
		if (!f.empty() && (si.width == 0 || si.height == 0)) {
			si.width  = f.cols;
			si.height = f.rows;
		}
		++frames;

		const double elapsed = now_ms_() - t0;
		if (elapsed >= measure_ms) {
			si.frames_counted = frames;
			si.fps_measured   = (elapsed > 0.0) ? (frames * 1000.0 / elapsed) : 0.0;
			break;
		}
	}

	if (si.frames_counted == 0 && si.fps_measured == 0.0) {
		si.fps_reported = cap_.get(cv::CAP_PROP_FPS);
	}

	return si;
}

video_capture::StreamInfo video_capture::negotiate(bool prefer_mjpg, int warmup_frames, int measure_ms) {
	if (!cap_.isOpened()) {
		if (!open_device_()) {
			std::cerr << "video_capture: failed to open camera " << cam_index_ << "\n";
			return info_;
		}
	}

	// Optionally set FOURCC to MJPG to unlock higher modes on many webcams.
	if (prefer_mjpg) {
		cap_.set(cv::CAP_PROP_FOURCC, static_cast<double>(cv::VideoWriter::fourcc('M','J','P','G')));
	}

	// Apply desired width/height/fps if provided
	if (desired_width_  > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH,  desired_width_);
	if (desired_height_ > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, desired_height_);
	if (desired_fps_    > 0.0) cap_.set(cv::CAP_PROP_FPS, desired_fps_);

	// Probe actuals
	info_ = probe_stream_(warmup_frames, measure_ms);

	// After probing, re-open stream for capture loop to start from a fresh state (optional but clean)
	cap_.release();
	if (!open_device_()) {
		std::cerr << "video_capture: failed to reopen camera after probe\n";
	}

	return info_;
}

bool video_capture::start_capture() {
	if (running_.load()) return true; // already running

	if (!cap_.isOpened()) {
		if (!open_device_()) {
			std::cerr << "video_capture: unable to open camera " << cam_index_ << "\n";
			return false;
		}
	}

	running_.store(true);
	th_ = std::thread(&video_capture::capture_loop_, this);
	return true;
}

void video_capture::stop_capture() {
	if (!running_.load()) return;
	running_.store(false);
	if (th_.joinable()) th_.join();
	if (cap_.isOpened()) cap_.release();
}

bool video_capture::snapshot_cpu(cv::Mat& dst) {
	std::lock_guard<std::mutex> lk(out_mutex_);
	if (out_cpu.empty()) return false;
	out_cpu.copyTo(dst);                 // deep copy
	return true;
}

bool video_capture::snapshot_gpu_download(cv::Mat& dst_cpu) {
	std::lock_guard<std::mutex> lk(out_mutex_);
	if (out_gpu.empty()) return false;
	out_gpu.download(dst_cpu);          // download while locked (no concurrent upload)
	return true;
}

// If you want a GPU→GPU deep copy later:
bool video_capture::snapshot_gpu(cv::cuda::GpuMat& dst_gpu) {
	std::lock_guard<std::mutex> lk(out_mutex_);
	if (out_gpu.empty()) return false;
	out_gpu.copyTo(dst_gpu);          // device-to-device copy
	return true;
}

void video_capture::capture_loop_() {
	// Simple capture loop:
	//  - Always grab into a CPU cv::Mat from cv::VideoCapture
	//  - If CUDA backend, upload to out_gpu; else keep in out_cpu
	//  - Tight loop—add sleep or timing control later if needed

	cv::Mat frame;

	while (running_.load()) {
		if (!cap_.read(frame)) {
			// Try to recover
			cap_.release();
			if (!open_device_()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}
			continue;
		}

		{
			std::lock_guard<std::mutex> lk(out_mutex_);
			if (backend_ == Backend::CUDA) {
				// Upload to GPU
				try {
					out_gpu.upload(frame);
				} catch (const cv::Exception& e) {
					// If OpenCV was built without CUDA, this throws; degrade to CPU
					// (You can remove this fallback if you prefer strict behavior.)
					// std::cerr << "CUDA upload failed, falling back to CPU: " << e.what() << "\n";
					backend_ = Backend::CPU;
					out_cpu = frame.clone();
				}
			} else {
				out_cpu = frame.clone();
			}
		}
	}
}

bool video_capture::open_device_preferring_(CapBackend pref) {
	// Try the preferred backend first, then the other one.
	auto try_open = [&](CapBackend b) -> bool {
#ifdef _WIN32
		int api = (b == CapBackend::MSMF) ? cv::CAP_MSMF : cv::CAP_DSHOW;
		if (cap_.open(cam_index_, api)) { opened_backend_ = b; return true; }
		return false;
#else
		(void)pref;
		if (cap_.open(cam_index_)) { opened_backend_ = CapBackend::Auto; return true; }
		return false;
#endif
	};

#ifdef _WIN32
	if (pref == CapBackend::MSMF) {
		if (try_open(CapBackend::MSMF)) return true;
		if (try_open(CapBackend::DSHOW)) return true;
	} else if (pref == CapBackend::DSHOW) {
		if (try_open(CapBackend::DSHOW)) return true;
		if (try_open(CapBackend::MSMF)) return true;
	} else
#endif
	{
		// Auto
#ifdef _WIN32
		if (try_open(CapBackend::MSMF)) return true;
		if (try_open(CapBackend::DSHOW)) return true;
#else
		return try_open(CapBackend::Auto);
#endif
	}
	return false;
}

bool video_capture::try_mode_(int w, int h, double fps, bool prefer_mjpg,
							int warmup_frames, int measure_ms, StreamInfo& out)
{
	cap_.release();
	if (!open_device_preferring_(CapBackend::MSMF)) { // prefer MSMF first
		// If MSMF failed entirely, fall back to the older path we had:
		if (!open_device_()) return false;
	}

	if (prefer_mjpg) {
		cap_.set(cv::CAP_PROP_FOURCC, static_cast<double>(cv::VideoWriter::fourcc('M','J','P','G')));
	}

	if (w  > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH,  w);
	if (h  > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, h);
	if (fps> 0) cap_.set(cv::CAP_PROP_FPS,          fps);

	// Probe what we actually got
	StreamInfo si = probe_stream_(warmup_frames, measure_ms);

	// Heuristic acceptance: size at least requested (or exact if requested), fps >= min threshold checked by caller
	out = si;
	return true;
}

video_capture::StreamInfo video_capture::negotiate_auto(int target_w, int target_h, double min_fps,
														bool prefer_mjpg, int measure_ms)
{
	// If caller set desired_* earlier, we’ll honor target_* here regardless.
	const int W = (target_w  > 0) ? target_w  : desired_width_;
	const int H = (target_h  > 0) ? target_h  : desired_height_;
	const double F = (desired_fps_ > 0.0) ? desired_fps_ : 60.0; // default “want 60”

	struct Cand { int w, h; double f; };
	std::vector<Cand> candidates;

	// 1) Exact target with common fps variants
	if (W>0 && H>0) {
		candidates.push_back({W, H, F});
		candidates.push_back({W, H, 60.0});
		candidates.push_back({W, H, 59.94});
		candidates.push_back({W, H, 30.0});
	}

	// 2) Common “big” fallbacks
	candidates.push_back({1920,1080,60.0});
	candidates.push_back({1920,1080,59.94});
	candidates.push_back({1920,1080,30.0});
	candidates.push_back({1280,720,60.0});
	candidates.push_back({1280,720,59.94});
	candidates.push_back({1280,720,30.0});

	// 3) Last-resort: let backend choose fps, just request size
	if (W>0 && H>0) candidates.push_back({W, H, 0.0});

	// Try all, keep the best that meets min_fps
	StreamInfo best{}; double best_score = -1.0;

	for (const auto& c : candidates) {
		StreamInfo si{};
		if (!try_mode_(c.w, c.h, c.f, prefer_mjpg, /*warmup=*/4, measure_ms, si)) {
			continue; // open failed
		}
		const double eff_fps = (si.fps_measured > 0.0 ? si.fps_measured : si.fps_reported);
		if (eff_fps < (min_fps - 0.5)) {
			// Misses fps requirement; skip unless nothing else works
			continue;
		}
		// Score = area * fps. Favor bigger + faster.
		const double area = static_cast<double>(si.width) * static_cast<double>(si.height);
		const double score = area * eff_fps;

		if (score > best_score) {
			best = si; best_score = score;
		}

		// Early exit if we hit exact W×H and fps >= min_fps
		if (W>0 && H>0 && si.width == W && si.height == H && eff_fps >= min_fps) {
			best = si; break;
		}
	}

	// If nothing met min_fps, take the best we measured anyway by relaxing fps
	if (best_score < 0.0) {
		for (const auto& c : candidates) {
			StreamInfo si{};
			if (!try_mode_(c.w, c.h, c.f, prefer_mjpg, /*warmup=*/3, measure_ms, si)) continue;
			const double eff_fps = (si.fps_measured > 0.0 ? si.fps_measured : si.fps_reported);
			const double area = static_cast<double>(si.width) * static_cast<double>(si.height);
			const double score = area * eff_fps;
			if (score > best_score) { best = si; best_score = score; }
		}
	}

	info_ = best;

	// Re-open clean for the capture loop with whichever backend succeeded last
	cap_.release();
	if (!open_device_preferring_(opened_backend_)) {
		// Last resort
		(void)open_device_();
	}

	// Lock in the chosen mode (best.width/height & an fps hint)
	if (prefer_mjpg) {
		cap_.set(cv::CAP_PROP_FOURCC, static_cast<double>(cv::VideoWriter::fourcc('M','J','P','G')));
	}
	if (best.width  > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH,  best.width);
	if (best.height > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, best.height);
	// Only set FPS if we have a confident value; some drivers ignore it anyway.
	if (best.fps_measured > 0.0) cap_.set(cv::CAP_PROP_FPS, best.fps_measured);

	return info_;
}

// --- public: selectVideoDevice() ---
// Pulls device list, shows a menu, validates up to 3 attempts, allows Q to quit.
int video_capture::selectVideoDevice() {
	auto devices = fetchVideoDevices_();
	if (devices.empty()) {
		std::cerr << "No video devices found.\n";
		return -1;
	}

	int tries = 0;
	while (tries < 3) {
		std::cout << "\nSelect a Video Capture Device\n";
		printOptions_(devices);
		std::cout << "Q - Quit\n";
		std::cout << "Enter device index: ";

		std::string input;
		if (!std::getline(std::cin, input)) {
			// handle potential leftover newline
			if (std::cin.fail()) {
				std::cin.clear();
			}
#ifdef _WIN32
			std::getline(std::cin, input);
#else
			return -1;
#endif
		}

		std::string lower = input;
		std::transform(lower.begin(), lower.end(), lower.begin(),
					[](unsigned char c){ return (char)std::tolower(c); });

		if (lower.find('q') != std::string::npos) return -1;

		try {
			int idx = std::stoi(input);
			auto it = std::find_if(devices.begin(), devices.end(),
								[idx](const VideoDevice& d){ return d.indexInList == idx; });
			if (it != devices.end()) {
				return idx; // success
			} else {
				std::cerr << "Index not in device list.\n";
			}
		} catch (...) {
			std::cerr << "Invalid index!\n";
		}

		clearOutput_();
		++tries;
	}
	return -1;
}

// --- private: fetchVideoDevices_() ---
std::vector<video_capture::VideoDevice> video_capture::fetchVideoDevices_() {
	std::vector<VideoDevice> list;

#ifndef _WIN32
	std::cerr << "Device listing is Windows-only (DirectShow). Skipping.\n";
	return list;
#else
	// Use FFmpeg to list DirectShow devices
	const std::string cmd = "ffmpeg -hide_banner -f dshow -list_devices true -i dummy";
	const std::string result = utils::runCommand(cmd);
	if (result.empty()) {
		std::cerr << "FFmpeg output empty (is ffmpeg on PATH?).\n";
		return list;
	}

	std::regex quoted_name_re(R"raw("([^"]+)")raw");

	int deviceLineCounter = 0;

	size_t start = 0;
	while (start < result.size()) {
		size_t end = result.find('\n', start);
		if (end == std::string::npos) end = result.size();

		std::string line = result.substr(start, end - start);
		std::string lower = line;
		std::transform(lower.begin(), lower.end(), lower.begin(),
					[](unsigned char c){ return (char)std::tolower(c); });

		std::smatch m;
		if (std::regex_search(line, m, quoted_name_re)) {
			const std::string deviceName = m[1].str();

			if (lower.find("video") != std::string::npos) {
				list.push_back(VideoDevice{ deviceLineCounter, deviceName });
				++deviceLineCounter;
			} else if (lower.find("audio") != std::string::npos) {
				++deviceLineCounter; // keep parity with FFmpeg's listing
			}
		}
		start = end + (end < result.size() ? 1 : 0);
	}

	return list;
#endif
}

// --- private: printOptions_() ---
void video_capture::printOptions_(const std::vector<VideoDevice>& list) {
	for (const auto& d : list) {
		std::cout << "  [" << d.indexInList << "] " << d.name << "\n";
	}
}

// --- private: clearOutput_() ---
void video_capture::clearOutput_() {
#ifdef _WIN32
	// Clear visible console buffer; scrollback behavior depends on console settings.
	system("cls");
#else
	// ANSI clear (commented out to keep scrollback in most terminals)
	// std::printf("\033[2J\033[H");
#endif
}