#include "apple_detection.hpp"
#include <iomanip>
#include <sstream>
#include <cmath>

using clock_steady = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using fsec = std::chrono::duration<float>;

// apple_track

// Add item up to capacity, remove oldest.
void apple::add(const cv::Point2f point, std::chrono::steady_clock::time_point time) {
	//std::lock_guard<std::mutex> lock(apple::apple_mtx_); // Thread safety
	PointInTime pit = PointInTime(point, time);
	// If track is at capacity
	if (track_.size() == capacity_) {
		track_.pop_back(); // Remove oldest
	}
	// Usin emplace to avoid copy/move
	track_.emplace_front(pit);
	lastPosition = pit;
	update_velocity_();
}

void apple::update_velocity_() {
	//std::lock_guard<std::mutex> lock(apple::apple_mtx_); // Thread safety
	using fsec = std::chrono::duration<float>;
	// Get the current position
	PointInTime curPIT = apple::lastPosition;
	PointInTime oldPIT = curPIT;
	constexpr float minDT = 1e-4f; // 0.1ms
	// We now need to find the next position
	// Filter using distance threshold to avoid jitter velocity.
	for (const PointInTime& pit : apple::track_) {
		// If distance is greater than the threshold
		if (dist_px(pit.point, oldPIT.point) > dist_thresh) {
			// update oldPIT
			oldPIT = pit;
			break;
		}
	}

	// Now that oldPIT has been updated, we can calculate the velocity.
	// Time difference
	float dt = std::chrono::duration_cast<fsec>(curPIT.time - oldPIT.time).count();
	// Distance as cv::Point2f
	cv::Point2f dp = curPIT.point - oldPIT.point;

	if (dt <= minDT) {
		velocity = {0, 0};
	} else {
		// Return the velocity vector in pixels/sec
		velocity = {dp.x / dt, dp.y / dt};
	}

}

// game_state

void game_state::clear_dets() {
	//std::lock_guard<std::mutex> lock(game_state::game_state_mtx_);
	game_state::dets_.points.clear();
}

int game_state::add_det(cv::Point2f point) {
	//std::lock_guard<std::mutex> lock(game_state::game_state_mtx_);
	game_state::dets_.points.push_back(point);
	return game_state::dets_.points.size();
}

std::string game_state::get_game_state_string() {
	//std::lock_guard<std::mutex> lock(game_state::game_state_mtx_);
	return std::string(apple_order.begin(), apple_order.end());
}

void game_state::update_apple_positions() {
	// Sanity check on apple_order size.
	if (apple_order.size() != 3) {
		return;
	}
	switch(apple_order[0]) {
		case 'S':
			sApple.add(lOrigin, clock_steady::now());
			break;
		case 'H':
			hApple.add(lOrigin, clock_steady::now());
			break;
		case 'A':
			aApple.add(lOrigin, clock_steady::now());
			break;
	}
	switch(apple_order[1]) {
		case 'S':
			sApple.add(cOrigin, clock_steady::now());
			break;
		case 'H':
			hApple.add(cOrigin, clock_steady::now());
			break;
		case 'A':
			aApple.add(cOrigin, clock_steady::now());
			break;
	}
	switch(apple_order[2]) {
		case 'S':
			sApple.add(rOrigin, clock_steady::now());
			break;
		case 'H':
			hApple.add(rOrigin, clock_steady::now());
			break;
		case 'A':
			aApple.add(rOrigin, clock_steady::now());
			break;
	}
	return;
}

// New methodology.
void game_state::update_game_state() {
	//std::lock_guard<std::mutex> lock(game_state::game_state_mtx_);
	// Make sure there are three dets
	if (dets_.points.size() != 3) {
		return;
	}
	// Get the current time.
	std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
	// Sort the list by x
	dets_.sort_by_x_then_y();
	// If we are not initialized
	if (!initialized_) {
		std::cout << "Initializing!\n";
		// Now update the apple positions and time.
		sApple.add(dets_.points[0], t);
		sApple.add(dets_.points[0], t);
		lOrigin = dets_.points[0];
		hApple.add(dets_.points[1], t);
		hApple.add(dets_.points[1], t);
		cOrigin = dets_.points[1];
		aApple.add(dets_.points[2], t);
		aApple.add(dets_.points[2], t);
		rOrigin = dets_.points[2];

		apple_order[0] = 'S';
		apple_order[1] = 'H';
		apple_order[2] = 'A';

		initialized_ = true;
		return;
	}
	// We are initialized

	// Before we get started
	// We need to make sure there is movement detected.
	// If none of the apples have moved, ther is no need to update the gamestate.
	bool lMove = true, cMove = true, rMove = true;
	for (cv::Point2f p : dets_.points) {
		// Calculate the distance for each apple
		float dist;
		if (dist_px(p, lOrigin) < dist_thresh && lMove) {
			lMove = false;
		}
		if (dist_px(p, cOrigin) < dist_thresh && cMove) {
			cMove = false;
		}
		if (dist_px(p, rOrigin) < dist_thresh && rMove) {
			rMove = false;
		}
	}

	// If no movement detected
	if (!lMove && !cMove && !rMove) {
		// This means all the apples are back to the origin.
		// If we have a pending swap, perform it here.
		int a = 0, b = 0;
		switch(centerOfRotation) {
			case 'L':
			case 'l':
				b = 1;
				centerOfRotation = char(0);
				break;
			case 'C':
			case 'c':
				b = 2;
				centerOfRotation = char(0);
				break;
			case 'R':
			case 'r':
				a = 1;
				b = 2;
				centerOfRotation = char(0);
				break;
		}
		if (a < apple_order.size() && b < apple_order.size() && a != b) {
			std::swap(apple_order[a], apple_order[b]);
			update_apple_positions();
		}
		return;
	}

	// Now we need to see which apples are moving
	// There will be a "center of rotation" either "L" "C" "R" or none.
	// There should only be two apples moving at a given time.
	// If two are moving, one is not. Find the one that is not moving.
	// Use this information to determine the center of rotation.
	if (lMove && cMove && !rMove) {
		centerOfRotation = 'L';
	} else if (lMove && !cMove && rMove) {
		centerOfRotation = 'C';
	} else if (!lMove && cMove && rMove) {
		centerOfRotation = 'R';
	} else {
		std::cout << "Unexpected movement! " << lMove << " " << cMove << " " << rMove << "\n";
	}

	return;

}

void game_state::reset() {
	// Update the apple order.
	apple_order[0] = 'S';
	apple_order[1] = 'H';
	apple_order[2] = 'A';
	initialized_ = false;
	lOrigin = {0.0f, 0.0f};
	cOrigin = {0.0f, 0.0f};
	rOrigin = {0.0f, 0.0f};
	return;
}

/*
void game_state::old_update_game_state() {
	//std::lock_guard<std::mutex> lock(game_state::game_state_mtx_);
	// Make sure there are three dets
	if (game_state::dets_.points.size() != 3) {
		return;
	}
	// Get the current time.
	std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
	// Sort the list by x
	game_state::dets_.sort_by_x_then_y();
	// If we are not initialized
	if (!game_state::initialized_) {
		// Now update the apple positions and time.
		game_state::sApple.add(game_state::dets_.points[0], t);
		game_state::hApple.add(game_state::dets_.points[1], t);
		game_state::aApple.add(game_state::dets_.points[2], t);

		std::cout << "Initializing!\n";

		game_state::g_state_string_ = "SHA";

		game_state::initialized_ = true;
		return;
	}
	// We are initialized

	// Before we get started
	// We need to make sure there is movement detected.
	// If none of the apples have moved, ther is no need to update the gamestate.
	bool sMove = true, hMove = true, aMove = true;
	for (cv::Point2f p : game_state::dets_.points) {
		// Calculate the distance for each apple
		float dist;
		if (dist_px(p, game_state::sApple.lastPosition.point) < dist_thresh && sMove) {
			sMove = false;
		}
		if (dist_px(p, game_state::hApple.lastPosition.point) < dist_thresh && hMove) {
			hMove = false;
		}
		if (dist_px(p, game_state::aApple.lastPosition.point) < dist_thresh && aMove) {
			aMove = false;
		}
	}

	// If no movement detected
	if (!sMove && !hMove && !aMove) {
		return;
	}

	// Now we have to figure out what apple is where.
	// To do this we need to run some calculations.
	// First is calculate the "predicted" position based off of apple velocity + position.
	// Because of the way the apples move, x+y velocity can be inconsistent. 
	// The apples move in v and ^ shapes. And suddenly swap y velocity on the apex.
	// However, the x velocity seems to remain constant. Weight the x precision higher than the y.

	cv::Point2f sPred, hPred, aPred;

	// Ideally dt should be the same for all three apples. Repeating this calculation is redundant.
	float dt = std::chrono::duration_cast<fsec>(t - sApple.lastPosition.time).count();
	sPred = sApple.lastPosition.point + sApple.velocity * dt;
	hPred = hApple.lastPosition.point + hApple.velocity * dt;
	aPred = aApple.lastPosition.point + aApple.velocity * dt;
	sApple.prediction = sPred;
	hApple.prediction = hPred;
	aApple.prediction = aPred;

	// Now that we have the predicted positions as a cv::Point2f
	// We need to determine what apple goes with the current positions.
	// As stated before the X velocity is most important here.
	// If the prediction using X only is unclear we can check the Y.
	// If Y does not clear things up, delay to the next gamestate.
	int method = 0;
	float s_delta = 0.0f, h_delta = 0.0f, a_delta = 0.0f, res = 0.0f;
	size_t sBest = -1,  hBest = -1, aBest = -1;

	// X prediction will be unclear if the dets x values are too similar.
	do {
		// Check to see if deltax is usable.
		// The points are in ascending x value, so absolute value is not important.
		if (dets_.points[1].x - dets_.points[0].x <= dx_thresh) {
			// Break out of the while loop and try distance check.
			break;
		}
		if (dets_.points[2].x - dets_.points[0].x <= dx_thresh) {
			// Break out of the while loop and try distance check.
			break;
		}
		if (dets_.points[2].x - dets_.points[1].x <= dx_thresh) {
			// Break out of the while loop and try distance check.
			break;
		}
		method = 1;
	} while(0);

	// 2D predictions will be unclear if apples are too close.
	do {
		if (method != 0) {
			break;
		}
		if (dist_px(dets_.points[0], dets_.points[1]) <= d2d_thresh) {
			break;
		}
		if (dist_px(dets_.points[0], dets_.points[2]) <= d2d_thresh) {
			break;
		}
		if (dist_px(dets_.points[1], dets_.points[2]) <= d2d_thresh) {
			break;
		}
		method = 2;
	} while (0);

	std::cout << "Using method " << method << "\n";

	switch (method) {
		case 1:
			s_delta = (std::abs(dets_.points[0].x - sPred.x));
			h_delta = (std::abs(dets_.points[0].x - hPred.x));
			a_delta = (std::abs(dets_.points[0].x - aPred.x));
			std::cout << "sD: " << s_delta << " hD: " << h_delta << " aD: " << a_delta << "\n";
			sBest = 0;
			hBest = 0;
			aBest = 0;
			for (size_t i = 1; i < dets_.points.size(); i++) {
				res = std::abs(dets_.points[i].x - sPred.x);
				s_delta = (res < s_delta) ? (sBest = i, res) : s_delta;
				res = std::abs(dets_.points[i].x - hPred.x);
				h_delta = (res < h_delta) ? (hBest = i, res) : h_delta;
				res = std::abs(dets_.points[i].x - aPred.x);
				a_delta = (res < a_delta) ? (aBest = i, res) : a_delta;
			}
			std::cout << "sD: " << s_delta << " hD: " << h_delta << " aD: " << a_delta << "\n";
			std::cout << "sB: " << sBest << " hB: " << hBest << " aB: " << aBest << "\n";
			break;
		case 2:
			s_delta = (dist_px(dets_.points[0], sPred));
			h_delta = (dist_px(dets_.points[0], hPred));
			a_delta = (dist_px(dets_.points[0], aPred));
			sBest = 0;
			hBest = 0;
			aBest = 0;
			for (size_t i = 1; i < dets_.points.size(); i++) {
				res = dist_px(dets_.points[i], sPred);
				s_delta = (res < s_delta) ? (sBest = i, res) : s_delta;
				res = dist_px(dets_.points[i], hPred);
				h_delta = (res < h_delta) ? (hBest = i, res) : h_delta;
				res = dist_px(dets_.points[i], aPred);
				a_delta = (res < a_delta) ? (aBest = i, res) : a_delta;
			}
			break;
		default:
			std::cout << "No valid Method\n";
			return;
	}

	// sanity check to make sure a value was asigned
	// Should never fail here
	if (sBest == -1 || hBest == -1 || aBest == -1) {
		std::cout << "Made it here :<\n";
		return;
	}

	// Sanity check to make sure none of the apples share a best.
	if (sBest == hBest || sBest == aBest || hBest == aBest) {
		std::cout << "Shared best not allowed!\n";
		return;
	}

	// By this point we are safe to make assignments.
	sApple.add(dets_.points[sBest], t);
	hApple.add(dets_.points[hBest], t);
	aApple.add(dets_.points[aBest], t);

}
*/

// apple_detection

// ---- ctor/dtor --------------------------------------------------------------

apple_detection::apple_detection(const Env& env, cv::Size input_size)
	: env_(env), input_size_(input_size) {
	update_diag_pre_(0.0);
	update_diag_fwd_(0.0);
	update_diag_post_(0.0);
}

apple_detection::~apple_detection() {
	stop();
}

// ---- lifecycle --------------------------------------------------------------

void apple_detection::start() {
	bool expected = false;
	if (!running_.compare_exchange_strong(expected, true)) return;

	t_pre_  = std::thread(&apple_detection::preprocess_loop_,  this);
	t_fwd_  = std::thread(&apple_detection::forward_loop_,     this);
	t_post_ = std::thread(&apple_detection::postprocess_loop_, this);
}

void apple_detection::stop() {
	if (!running_.exchange(false)) return;
	{
		std::lock_guard<std::mutex> lk(mtx_);
		// wake any waits
	}
	cv_.notify_all();

	if (t_pre_.joinable())  t_pre_.join();
	if (t_fwd_.joinable())  t_fwd_.join();
	if (t_post_.joinable()) t_post_.join();
}

// ---- model path -------------------------------------------------------------

void apple_detection::set_model_path(const std::string& p) {
	std::lock_guard<std::mutex> lk(net_mtx_);
	model_path_ = p;
	net_ = cv::dnn::Net();
	net_loaded_ = false;
}

std::string apple_detection::model_path() const {
	std::lock_guard<std::mutex> lk(net_mtx_);
	return model_path_;
}

// ---- ingress ---------------------------------------------------------------

void apple_detection::push_cpu(const cv::Mat& bgr) {
	if (bgr.empty()) return;
	{
		std::lock_guard<std::mutex> lk(mtx_);
		in_cpu_      = bgr.clone();
		in_has_gpu_  = false;
		in_has_new_  = true;
	}
	cv_.notify_all();
}

#ifdef HAVE_OPENCV_CUDAWARPING
void apple_detection::push_gpu(const cv::cuda::GpuMat& bgr) {
	if (bgr.empty()) return;
	{
		std::lock_guard<std::mutex> lk(mtx_);
		in_gpu_      = bgr.clone();
		in_has_gpu_  = true;
		in_has_new_  = true;
	}
	cv_.notify_all();
}
#endif

// ---- pre→fwd pop (optional external use) -----------------------------------

bool apple_detection::pop_preprocessed(FwdItem& out, int timeout_ms) {
	std::unique_lock<std::mutex> lk(mtx_);
	if (timeout_ms < 0) {
		cv_.wait(lk, [&] { return !running_ || fwd_ready_; });
	} else {
		auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
		if (!cv_.wait_until(lk, until, [&] { return !running_ || fwd_ready_; }))
			return false;
	}
	if (!running_ && !fwd_ready_) return false;

	out = std::move(fwd_item_);
	fwd_item_ = FwdItem{};
	fwd_ready_ = false;

	lk.unlock();
	cv_.notify_all();
	return true;
}

// ---- config -----------------------------------------------------------------

void apple_detection::set_input_size(cv::Size sz) {
	if (sz.width > 0 && sz.height > 0) input_size_ = sz;
}

// ---- diagnostics ------------------------------------------------------------

std::string apple_detection::diagnostic() const {
	std::lock_guard<std::mutex> lk(diag_mtx_);
	return diag_;
}

std::string apple_detection::diagnostic_2() const {
	std::lock_guard<std::mutex> lk(diag_2_mtx_);
	return diag_2_;
}

void apple_detection::update_diag_pre_(double ms) {
	std::lock_guard<std::mutex> lk(diag_mtx_);
	pre_ms_ = ms;
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2)
		<< "pre="  << pre_ms_  << "ms "
		<< "fwd="  << (fwd_ms_)  << "ms "
		<< "post=" << (post_ms_) << "ms";
	diag_ = oss.str();
}

void apple_detection::update_diag_fwd_(double ms) {
	std::lock_guard<std::mutex> lk(diag_mtx_);
	fwd_ms_ = ms;
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2)
		<< "pre="  << pre_ms_  << "ms "
		<< "fwd="  << fwd_ms_  << "ms "
		<< "post=" << post_ms_ << "ms";
	diag_ = oss.str();
}

void apple_detection::update_diag_post_(double ms) {
	std::lock_guard<std::mutex> lk(diag_mtx_);
	post_ms_ = ms;
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2)
		<< "pre="  << pre_ms_  << "ms "
		<< "fwd="  << fwd_ms_  << "ms "
		<< "post=" << post_ms_ << "ms";
	diag_ = oss.str();
}

void apple_detection::update_diag_dets_(int dets) {
	std::lock_guard<std::mutex> lk(diag_2_mtx_);
	diag_dets_ = dets;
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2)
		<< "dets=" << diag_dets_
		<< "state=" << g_state_string_;
	diag_2_ = oss.str();
}

std::string apple_detection::get_game_state_string() {
	return g_state_.get_game_state_string();
}

void apple_detection::reset_game_state() {
	g_state_.reset();
}

void apple_detection::update_diag_order_() {
	std::lock_guard<std::mutex> lk(diag_2_mtx_);
	std::string g_state_string_ = get_game_state_string();
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2)
		<< "dets=" << diag_dets_
		<< "state=" << g_state_string_;
	diag_2_ = oss.str();
}

cv::Mat apple_detection::get_annotated_clone() const {
	std::lock_guard<std::mutex> lk(annotated_mtx_);
	return annotated.clone();
}

// ---- model loader (lazy) ----------------------------------------------------

bool apple_detection::load_if_needed_() {
	std::lock_guard<std::mutex> lk(net_mtx_);
	if (net_loaded_) return true;
	if (model_path_.empty()) return false;

	try {
		net_ = cv::dnn::readNet(model_path_);
	} catch (...) {
		net_ = cv::dnn::Net();
		net_loaded_ = false;
		return false;
	}
	if (net_.empty()) {
		net_loaded_ = false;
		return false;
	}

	try {
		if (env_.backend == Backend::CUDA) {
#if CV_VERSION_MAJOR>=4
			net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			if (env_.prefer_fp16)
				net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
			else
				net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
			net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
		} else {
			net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net_.setPreferableTarget (cv::dnn::DNN_TARGET_CPU);
		}
	} catch (...) {
		// Fallback to CPU if CUDA not available in this build
		net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net_.setPreferableTarget (cv::dnn::DNN_TARGET_CPU);
	}

	net_loaded_ = true;
	return true;
}

// ---- pre-process thread -----------------------------------------------------

void apple_detection::preprocess_loop_() {
	while (running_) {
		// Wait for new input and free handoff
		cv::Mat in_local_cpu;
#ifdef HAVE_OPENCV_CUDAWARPING
		cv::cuda::GpuMat in_local_gpu;
#endif
		bool use_gpu = false;

		{
			std::unique_lock<std::mutex> lk(mtx_);
			cv_.wait(lk, [&] {
				return !running_ || (in_has_new_ && !fwd_ready_);
			});
			if (!running_) break;

			if (in_has_gpu_) {
#ifdef HAVE_OPENCV_CUDAWARPING
				in_local_gpu = std::move(in_gpu_);
				in_gpu_.release();
				use_gpu = (env_.backend == Backend::CUDA && !in_local_gpu.empty());
#else
				in_local_cpu = std::move(in_cpu_);
				in_cpu_.release();
				use_gpu = false;
#endif
			} else {
				in_local_cpu = std::move(in_cpu_);
				in_cpu_.release();
				use_gpu = false;
			}
			in_has_new_ = false;
		}

		auto t0 = clock_steady::now();

		// --- compute letterbox geometry ---
		const int inW = input_size_.width;
		const int inH = input_size_.height;

		cv::Mat original_bgr;
		cv::Size orig_size;
		cv::Mat model_input_bgr;  // the 480x480 (letterboxed) image fed to blob

		#ifdef HAVE_OPENCV_CUDAWARPING
		if (use_gpu) {
			in_local_gpu.download(original_bgr);
			orig_size = original_bgr.size();
		} else
		#endif
		{
			original_bgr = in_local_cpu;
			orig_size    = original_bgr.size();
		}

		if (original_bgr.empty()) {
			// publish empty and continue
			std::lock_guard<std::mutex> lk(mtx_);
			fwd_item_ = FwdItem{};
			fwd_ready_ = true;
			cv_.notify_all();
			update_diag_pre_(0.0);
			continue;
		}

		float scale = 1.f;
		int pad_x = 0, pad_y = 0;

		if (!letterbox_) {
			// old behavior: direct resize (may stretch)
			if (orig_size != input_size_) {
				cv::resize(original_bgr, model_input_bgr, input_size_, 0, 0, cv::INTER_LINEAR);
			} else {
				model_input_bgr = original_bgr;
			}
			scale = static_cast<float>(inW) / std::max(1, orig_size.width); // rough, unused if not letterboxing
			pad_x = pad_y = 0;
		} else {
			// Keep aspect ratio, pad with 114 (YOLO convention)
			float s = std::min(inW / static_cast<float>(orig_size.width),
							inH / static_cast<float>(orig_size.height));
			int newW = std::max(1, static_cast<int>(std::round(orig_size.width  * s)));
			int newH = std::max(1, static_cast<int>(std::round(orig_size.height * s)));

			cv::Mat resized;
		#ifdef HAVE_OPENCV_CUDAWARPING
			if (use_gpu) {
				cv::cuda::GpuMat gpu_resized;
				cv::cuda::resize(in_local_gpu, gpu_resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
				gpu_resized.download(resized);
			} else
		#endif
			{
				cv::resize(original_bgr, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
			}

			// Pad into a fixed canvas
			model_input_bgr = cv::Mat(inH, inW, CV_8UC3, cv::Scalar(114,114,114));
			pad_x = (inW - newW) / 2;
			pad_y = (inH - newH) / 2;
			resized.copyTo(model_input_bgr(cv::Rect(pad_x, pad_y, newW, newH)));

			scale = s;
		}

		// Make the blob from the letterboxed (or resized) image
		cv::Mat blob;
		cv::dnn::blobFromImage(
			model_input_bgr, blob,
			scalefactor_, input_size_, mean_, swapRB_, /*crop*/ false, CV_32F
		);

		// publish
		{
			std::lock_guard<std::mutex> lk(mtx_);
			fwd_item_.blob         = std::move(blob);
			fwd_item_.original_bgr = std::move(original_bgr);
			fwd_item_.orig_size    = orig_size;
			fwd_item_.seq          = seq_.fetch_add(1, std::memory_order_relaxed);
			fwd_item_.lb_scale     = scale;
			fwd_item_.lb_pad_x     = pad_x;
			fwd_item_.lb_pad_y     = pad_y;
			fwd_ready_             = true;
		}
		cv_.notify_all();

		auto t1 = clock_steady::now();
		double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
		update_diag_pre_(ms);
	}
}

// ---- forward (inference) thread --------------------------------------------

void apple_detection::forward_loop_() {
	while (running_) {
		// Wait for a preprocessed item
		FwdItem item;
		if (!pop_preprocessed(item, /*timeout_ms*/ 100)) {
			if (!running_) break;
			continue;
		}
		if (!load_if_needed_()) {
			// No model yet; drop silently to avoid blocking the pipeline
			continue;
		}

		auto t0 = clock_steady::now();

		// Set input & run forward
		if (!item.blob.empty()) {
			try {
				net_.setInput(item.blob);
			} catch (...) {
				// Skip this frame if setInput fails
				continue;
			}
		} else {
			continue; // nothing to infer on
		}

		std::vector<cv::Mat> outs;
		try {
			// Try unconnected layer names first (covers most ONNX exports)
			std::vector<std::string> outNames = net_.getUnconnectedOutLayersNames();
			if (!outNames.empty()) {
				net_.forward(outs, outNames);
			} else {
				cv::Mat out = net_.forward();
				outs.clear();
				outs.emplace_back(std::move(out));
			}
		} catch (...) {
			// forward failed; skip
			continue;
		}

		auto t1 = clock_steady::now();
		double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
		update_diag_fwd_(ms);

		// Wait until post slot is free, then publish
		{
			std::unique_lock<std::mutex> lk(mtx_);
			cv_.wait(lk, [&] { return !running_ || !post_ready_; });
			if (!running_) break;

			post_item_.outs        = std::move(outs);
			post_item_.original_bgr= std::move(item.original_bgr);
			post_item_.orig_size   = item.orig_size;
			post_item_.seq         = item.seq;
			post_ready_            = true;

			// NEW: carry letterbox info
			post_item_.lb_scale     = item.lb_scale;
			post_item_.lb_pad_x     = item.lb_pad_x;
			post_item_.lb_pad_y     = item.lb_pad_y;
		}
		cv_.notify_all();
	}
}

// ---- post-process thread ----------------------------------------------------

void apple_detection::drawCharAbove(cv::Mat& img, char ch, const cv::Point2f& p,
	const cv::Scalar& color, int fontFace,
	double fontScale, int thickness, int offsetY) // pixels above the point
{
	std::string s(1, ch);

	int baseline = 0;
	cv::Size ts = cv::getTextSize(s, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;

	int baseY = static_cast<int>(std::lround(p.y)) - offsetY;
	int leftX = static_cast<int>(std::lround(p.x)) - ts.width / 2;

	// outline (stroke) + fill for readability
	cv::putText(img, s, {leftX, baseY}, fontFace, fontScale, {0,0,0}, thickness+2, cv::LINE_AA);
	cv::putText(img, s, {leftX, baseY}, fontFace, fontScale, color,          thickness,   cv::LINE_AA);

	// optional: mark the point
	// cv::circle(img, p, 3, {0,255,255}, cv::FILLED, cv::LINE_AA);
}

static inline cv::Rect clampRect(const cv::Rect& r, const cv::Size& im) {
	int x = std::max(0, r.x);
	int y = std::max(0, r.y);
	int w = std::min(r.width,  im.width  - x);
	int h = std::min(r.height, im.height - y);
	if (w < 0) w = 0; if (h < 0) h = 0;
	return {x, y, w, h};
}

void apple_detection::postprocess_loop_() {
	while (running_) {
		PostItem item;
		{
			std::unique_lock<std::mutex> lk(mtx_);
			cv_.wait(lk, [&] { return !running_ || post_ready_; });
			if (!running_) break;

			item = std::move(post_item_);
			post_item_ = PostItem{};
			post_ready_ = false;
		}
		cv_.notify_all();

		auto t0 = clock_steady::now();

		cv::Mat vis = item.original_bgr; // work directly on the original
		if (vis.empty()) {
			// No frame at all → publish nothing
			std::lock_guard<std::mutex> lk(annotated_mtx_);
			annotated.release();
			continue;
		}

		// If there are no model outputs yet, still show the raw frame.
		if (item.outs.empty()) {
			std::lock_guard<std::mutex> lk(annotated_mtx_);
			annotated = vis;
			// still update post timing below
			auto t1 = clock_steady::now();
			double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
			update_diag_post_(ms);
			continue;
		}

		// Parse YOLOv5/8/11-like head output robustly.
		// We normalize two common layouts:
		//  - [1, N, C]  OR  [1, C, N]
		//  - where C = 4 (+obj?) + num_classes
		// After normalization, we iterate rows as predictions.
		cv::Mat out = item.outs[0];
		cv::Mat preds;  // 2D float: rows=predictions, cols=channels

		if (out.dims == 3) {
			int b = out.size[0]; (void)b; // usually 1
			int d1 = out.size[1];
			int d2 = out.size[2];
			if (d1 < d2) {
				// [1, C, N] -> transpose to [N, C]
				cv::Mat tmp(d1, d2, CV_32F, out.ptr<float>());
				preds = tmp.t(); // [N,C]
			} else {
				// [1, N, C] -> make [N, C]
				preds = cv::Mat(d1, d2, CV_32F, out.ptr<float>()).clone();
			}
		} else if (out.dims == 2) {
			// Already 2D
			preds = out.clone();
		} else {
			// Unexpected; try flatten best-effort assuming C last
			preds = out.reshape(1, out.total() / out.size[out.dims - 1]).clone();
		}

		const int num_preds = preds.rows;
		const int ch        = preds.cols;
		if (num_preds <= 0 || ch < 5) {
			// Not a YOLO-like tensor; publish original unchanged
			std::lock_guard<std::mutex> lk(annotated_mtx_);
			annotated = vis;
			continue;
		}

		// Heuristic: check if tensor includes objectness at channel 4
		// If values around channel 4 look like confidences we keep it; otherwise treat as no obj.
		bool has_obj = (ch >= 6); // common case

		// Gather boxes/scores
		std::vector<cv::Rect>  boxes;
		std::vector<float>     scores;
		std::vector<int>       class_ids;
		boxes.reserve(num_preds); scores.reserve(num_preds); class_ids.reserve(num_preds);

		const int inW = input_size_.width;
		const int inH = input_size_.height;
		const float sx  = item.orig_size.width  / inW;
		const float sy  = item.orig_size.height / inH;

		for (int i = 0; i < num_preds; ++i) {
			const float* p = preds.ptr<float>(i);
			float x = p[0], y = p[1], w = p[2], h = p[3];

			int cls_start = has_obj ? 5 : 4;
			int num_cls   = ch - cls_start;
			if (num_cls <= 0) continue;

			// Find top class score
			int best_c = 0;
			float best_s = -1.f;
			for (int c = 0; c < num_cls; ++c) {
				float s = p[cls_start + c];
				if (s > best_s) { best_s = s; best_c = c; }
			}
			float obj = has_obj ? p[4] : 1.0f;
			float conf = obj * best_s;

			if (conf < env_.detector_conf) continue;

			// If normalized to [0,1], scale to model-input size first
			if (w <= 2.0f && h <= 2.0f && x <= 1.5f && y <= 1.5f) {
				x *= inW; y *= inH; w *= inW; h *= inH;
			}

			// Undo letterbox padding and scale back to original image
			const float scale = item.lb_scale; // we need these from the same seq
			const int   px    = item.lb_pad_x;
			const int   py    = item.lb_pad_y;

			// Model outputs are in the model-input (letterboxed) space.
			// Convert center -> tl, remove pad, then divide by scale.
			float cx = (x - px) / scale;
			float cy = (y - py) / scale;
			float bw0 = w / scale;
			float bh0 = h / scale;

			float left = cx - 0.5f * bw0;
			float top  = cy - 0.5f * bh0;
			float bw   = bw0;
			float bh   = bh0;

			cv::Rect rect(
				static_cast<int>(std::round(left)),
				static_cast<int>(std::round(top)),
				static_cast<int>(std::round(bw)),
				static_cast<int>(std::round(bh))
			);

			rect = clampRect(rect, item.orig_size);
			if (rect.width <= 0 || rect.height <= 0) continue;

			boxes.push_back(rect);
			scores.push_back(conf);
			class_ids.push_back(best_c); // kept for future use (apple class = 0)
		}

		// NMS
		std::vector<int> keep;
		cv::dnn::NMSBoxes(boxes, scores, env_.detector_conf, env_.detector_nms, keep);

		// Reset the detection list.
		apple_detection::g_state_.clear_dets();
		int dets = 0;

		// Draw
		for (int idx : keep) {
			const cv::Rect& r = boxes[idx];
			cv::rectangle(vis, r, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

			// Blue dot at center of mass (center of the rect)
			cv::Point2f center(r.x + r.width/2, r.y + r.height/2);
			cv::circle(vis, center, 3, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);

			dets = apple_detection::g_state_.add_det(center);
		}

		update_diag_dets_(dets);
		g_state_.update_game_state();

		drawCharAbove(vis, g_state_.sApple.name, g_state_.sApple.lastPosition.point);
		drawCharAbove(vis, g_state_.hApple.name, g_state_.hApple.lastPosition.point);
		drawCharAbove(vis, g_state_.aApple.name, g_state_.aApple.lastPosition.point);

		// Draw origin circles.
		cv::circle(vis, g_state_.lOrigin, dist_thresh ,cv::Scalar(255, 0, 0), 1);
		cv::circle(vis, g_state_.cOrigin, dist_thresh ,cv::Scalar(255, 0, 0), 1);
		cv::circle(vis, g_state_.rOrigin, dist_thresh ,cv::Scalar(255, 0, 0), 1);

		/*
		cv::line(vis, g_state_.sApple.lastPosition.point, g_state_.sApple.prediction, cv::Scalar(255, 0, 0), 4);
		cv::line(vis, g_state_.hApple.lastPosition.point, g_state_.hApple.prediction, cv::Scalar(255, 0, 0), 4);
		cv::line(vis, g_state_.aApple.lastPosition.point, g_state_.aApple.prediction, cv::Scalar(255, 0, 0), 4);
		*/

		// Publish annotated frame
		{
			std::lock_guard<std::mutex> lk(annotated_mtx_);
			annotated = vis; // note: vis is item.original_bgr which we own; shallow copy is fine
		}

		auto t1 = clock_steady::now();
		double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
		update_diag_post_(ms);
	}
}