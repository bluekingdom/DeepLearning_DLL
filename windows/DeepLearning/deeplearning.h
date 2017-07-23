#pragma once

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#ifdef CAFFE_DLL_EXPORTS
#define CAFFE_DLL_API __declspec(dllexport)
#else
#define CAFFE_DLL_API __declspec(dllimport)
#endif

using std::string;
using std::vector;

class CAFFE_DLL_API DeepLearning
{
public:
	struct Detect_Result
	{
		int id;
		int label;
		float score;
		cv::Vec4f bbox;
	};
	struct Detect_Result_Batch
	{
		std::vector<Detect_Result> vDetectRst;
	};

	typedef Detect_Result SSD_Result;
	typedef Detect_Result_Batch SSD_Result_Batch;

	struct Classify_Result
	{
		std::vector<int> vOutput;
	};

public:
	DeepLearning();
	~DeepLearning();

	bool Init( const string& model_file, const std::vector<int>& mean_value, const string& weights_file, bool is_gpu = false );
	bool Init( const string& model_file, const string& mean_file, const string& weights_file, bool is_gpu = false );
	bool Init_FRCNN( const string& model_file, const string& config_file, const string& weights_file, 
		int gpu_idx = -1 );

	bool SetScale( float scale );

	bool SetDevice( int gpu_id );
	static void SetBlasThreadNum( int thread_num );

	bool Release();
	bool Release_FRCNN();

	bool Detect_SSD( const cv::Mat& img, std::vector<Detect_Result>& detections );
	bool Detect_SSD_Batch( const std::vector<cv::Mat>& imgs, std::vector<Detect_Result_Batch>& detections );
	bool Classify( const cv::Mat& img, std::vector<int>& outputs );
	bool Classify_Batch( const std::vector<cv::Mat>& imgs, std::vector<Classify_Result>& results);

	bool Detect_FRCNN( const cv::Mat& img, std::vector<Detect_Result>& detections );

	const cv::Size& GetInputSize() const;

private:

	bool NetworkForward( const cv::Mat& img );
	bool NetworkForward_Batch( const std::vector<cv::Mat>& imgs );

	void SetMean( const std::vector<int>& mean_value );
	void SetMean( const string& mean_file );

	void WrapInputLayer( std::vector<cv::Mat>* input_channels );
	void WrapInputLayer_Batch( std::vector<std::vector<cv::Mat>>& input_channels );

	void Preprocess( const cv::Mat& img, std::vector<cv::Mat>* input_channels );

	bool LoadModel( const string& model_file, const string& weights_file, bool is_gpu = false );

private:
	void *m_network;
	cv::Size m_inputGeometry;
	int m_iNumChannels;
	cv::Mat m_mean;
	float m_fScale;
};