#include "apple_detection.hpp"
#include <iomanip>
#include <sstream>

using clock_steady = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

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

		cv::Mat original_bgr;
		cv::Size orig_size;
		cv::Mat resized_cpu;

#ifdef HAVE_OPENCV_CUDAWARPING
		if (use_gpu) {
			if (!in_local_gpu.empty()) {
				in_local_gpu.download(original_bgr);
				orig_size = original_bgr.size();

				cv::cuda::GpuMat resized_gpu;
				cv::cuda::resize(in_local_gpu, resized_gpu, input_size_, 0, 0, cv::INTER_LINEAR);
				resized_gpu.download(resized_cpu);
			}
		} else
#endif
		{
			original_bgr = in_local_cpu;
			orig_size    = original_bgr.size();

			if (!original_bgr.empty() && input_size_ != original_bgr.size()) {
				cv::resize(original_bgr, resized_cpu, input_size_, 0, 0, cv::INTER_LINEAR);
			} else {
				resized_cpu = original_bgr;
			}
		}

		cv::Mat blob;
		if (!resized_cpu.empty()) {
			cv::dnn::blobFromImage(
				resized_cpu,
				blob,
				scalefactor_,
				input_size_,
				mean_,
				swapRB_,
				/*crop*/ false,
				CV_32F
			);
		}

		{
			std::lock_guard<std::mutex> lk(mtx_);
			fwd_item_.blob         = std::move(blob);
			fwd_item_.original_bgr = std::move(original_bgr);
			fwd_item_.orig_size    = orig_size;
			fwd_item_.seq          = seq_.fetch_add(1, std::memory_order_relaxed);
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
		}
		cv_.notify_all();
	}
}

// ---- post-process thread ----------------------------------------------------

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

		const float inW = static_cast<float>(input_size_.width);
		const float inH = static_cast<float>(input_size_.height);
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

			if (conf < conf_thresh_) continue;

			// If normalized to [0,1], scale to input size
			if (w <= 2.0f && h <= 2.0f && x <= 1.5f && y <= 1.5f) {
				x *= inW; y *= inH; w *= inW; h *= inH;
			}

			// Convert center x,y,w,h to top-left box in original image coords
			float left = (x - 0.5f * w) * sx;
			float top  = (y - 0.5f * h) * sy;
			float bw   = w * sx;
			float bh   = h * sy;

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
		cv::dnn::NMSBoxes(boxes, scores, conf_thresh_, nms_thresh_, keep);

		// Draw
		for (int idx : keep) {
			const cv::Rect& r = boxes[idx];
			cv::rectangle(vis, r, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

			// Blue dot at center of mass (center of the rect)
			cv::Point center(r.x + r.width/2, r.y + r.height/2);
			cv::circle(vis, center, 3, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);
		}

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