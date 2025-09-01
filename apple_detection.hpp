#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef HAVE_OPENCV_CUDAWARPING
#include <opencv2/cudawarping.hpp>
#endif

class apple_detection {
public:
	// Environment selection for this detector
	enum class Backend { CPU, CUDA };
	struct Env {
		Backend backend = Backend::CPU;
		bool    prefer_fp16 = false;   // used by forward thread for DNN target
		float   detector_conf = 0.70f;
		float   detector_nms  = 0.70f;
	};

	// Item passed from pre-process → forward
	struct FwdItem {
		cv::Mat blob;
		cv::Mat original_bgr;
		cv::Size orig_size{};
		int64_t  seq = 0;

		// NEW: letterbox mapping (input->original)
		float lb_scale = 1.f;  // scale applied to original before padding
		int   lb_pad_x = 0;    // left padding in model-input coords
		int   lb_pad_y = 0;    // top  padding in model-input coords
	};


	// Item passed from forward → post-process
	struct PostItem {
		std::vector<cv::Mat> outs; // raw outputs from net.forward()
		cv::Mat original_bgr;      // original frame
		cv::Size orig_size{};
		int64_t  seq = 0;

		// NEW: letterbox info from FwdItem
		float lb_scale = 1.f;
		int   lb_pad_x = 0;
		int   lb_pad_y = 0;
	};

	explicit apple_detection(const Env& env = {}, cv::Size input_size = {640, 640});
	~apple_detection();

	// Lifecycle
	void start();   // starts pre-process, forward, and post-process threads
	void stop();
	bool is_running() const { return running_.load(); }

	// Capture → detector ingress (call from your video_capture bridge)
	void push_cpu(const cv::Mat& bgr);
#ifdef HAVE_OPENCV_CUDAWARPING
	void push_gpu(const cv::cuda::GpuMat& bgr);
#endif

	// Optional: forward loop can be externalized, but we keep it internal.
	// Exposed pop in case you later want to inject a custom forward stage.
	bool pop_preprocessed(FwdItem& out, int timeout_ms = -1);

	// Configuration
	void set_input_size(cv::Size sz);
	cv::Size input_size() const { return input_size_; }
	void set_env(const Env& e)  { env_ = e; }

	void set_model_path(const std::string& p);
	std::string model_path() const;

	void set_conf_threshold(float t) { env_.detector_conf = t; }
	void set_nms_threshold(float t)  { env_.detector_nms  = t; }

	void set_letterbox(bool on) { letterbox_ = on; }

	// Diagnostics (updated by threads; not displayed here)
	std::string diagnostic() const;   // e.g., "pre=1.6ms fwd=22.4ms post=2.1ms"
	std::string diagnostic_2() const; // e.g., "dets=3"

	// Output for your display bridge (reader usually clones this)
	cv::Mat annotated;
	// Thread-safe getter (preferred if you can call this in your display thread)
	cv::Mat get_annotated_clone() const;

private:
	// --- Threads ---
	void preprocess_loop_();
	void forward_loop_();
	void postprocess_loop_();

	// --- Helpers ---
	void update_diag_pre_(double ms);
	void update_diag_fwd_(double ms);
	void update_diag_post_(double ms);
	void update_diag_dets_(int dets);
	bool load_if_needed_(); // lazy model load inside forward loop

	// --- Config ---
	Env       env_;
	cv::Size  input_size_;
	float     scalefactor_ = 1.f / 255.f;     // YOLO-style normalization
	bool      swapRB_      = true;
	cv::Scalar mean_       = cv::Scalar();    // (0,0,0)
	bool      letterbox_   = true;

	// --- State & sync ---
	std::atomic<bool> running_{false};
	std::thread       t_pre_;
	std::thread       t_fwd_;
	std::thread       t_post_;

	mutable std::mutex mtx_;
	std::condition_variable cv_;

	// Ingress: "latest input" single-slot (CPU or GPU). Overwritten by push_*.
	cv::Mat            in_cpu_;
#ifdef HAVE_OPENCV_CUDAWARPING
	cv::cuda::GpuMat   in_gpu_;
#endif
	bool               in_has_gpu_ = false;
	bool               in_has_new_ = false;

	// Pre→Fwd: single-slot handoff. Pre blocks while this is occupied.
	FwdItem            fwd_item_;
	bool               fwd_ready_ = false;

	// Fwd→Post: single-slot handoff. Fwd blocks while this is occupied.
	PostItem           post_item_;
	bool               post_ready_ = false;

	// Model
	mutable std::mutex net_mtx_;
	cv::dnn::Net       net_;
	bool               net_loaded_ = false;
	std::string        model_path_;

	// Diagnostics
	mutable std::mutex diag_mtx_;
	double pre_ms_  = 0.0;
	double fwd_ms_  = 0.0;
	double post_ms_ = 0.0;
	std::string diag_;
	mutable std::mutex diag_2_mtx_;
	int diag_dets_ = 0;
	std::string diag_2_;

	// Sequencing
	std::atomic<int64_t> seq_{0};

	// Annotated frame guard
	mutable std::mutex annotated_mtx_;
};