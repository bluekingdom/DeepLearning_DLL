
//#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "deeplearning.h"
#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "caffe/FRCNN/api.hpp"

using namespace caffe;
#define NET_TYPE(NAME) static_cast<Net<float>*>(##NAME)

#define FRCNN_NET_TYPE(NAME) static_cast<FRCNN_API::Detector*>(##NAME)

DeepLearning::DeepLearning()
: m_network(nullptr )
, m_inputGeometry(0, 0 )
, m_iNumChannels( 0 )
, m_fScale( 1.f )
{
	//FLAGS_minloglevel = 3;
}

DeepLearning::~DeepLearning() {
	//Release();
}

bool DeepLearning::Init(const string& model_file,
	const std::vector<int>& mean_value,
	const string& weights_file,
	bool is_gpu) {

	if ( false == LoadModel( model_file, weights_file, is_gpu ) )
		return false;
	SetMean(mean_value);

	return true;
}

bool DeepLearning::Init( const string& model_file, const string& mean_file, const string& weights_file, bool is_gpu /*= false */ )
{
	if ( false == LoadModel( model_file, weights_file, is_gpu ) )
		return false;
	SetMean(mean_file);

	return true;
}

bool DeepLearning::LoadModel( const string& model_file, const string& weights_file, bool is_gpu /*= false */ )
{
	if ( is_gpu == true )
	{
		Caffe::set_mode( Caffe::GPU );
	}
	else
	{
		Caffe::set_mode( Caffe::CPU );
	}

	/* Load the network. */
	//	NET_TYPE(net_ssd)->reset(new Net<float>(model_file, TEST));
	m_network = new Net<float>(model_file, TEST);
	NET_TYPE(m_network)->CopyTrainedLayersFrom(weights_file);

	CHECK_EQ(NET_TYPE(m_network)->num_inputs(), 1) << "Network should have exactly one input.";
	//CHECK_EQ(NET_TYPE(m_network)->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = NET_TYPE(m_network)->input_blobs()[0];
	m_iNumChannels = input_layer->channels();
	CHECK(m_iNumChannels == 3 || m_iNumChannels == 1)
		<< "Input layer should have 1 or 3 channels.";
	m_inputGeometry = cv::Size(input_layer->width(), input_layer->height());

	SetScale( 1.0f );

	return true;
}

bool DeepLearning::Release() {
	if ( m_network )
	{
		delete NET_TYPE( m_network );
	}
	m_network = nullptr;
	return true;
}

/* Load the mean file in binaryproto format. */
void DeepLearning::SetMean(const std::vector<int>& mean_value) {
	cv::Scalar channel_mean;
	if (!mean_value.empty()) {
		CHECK(mean_value.size() == 1 || mean_value.size() == m_iNumChannels) <<
			"Specify either 1 mean_value or as many as channels: " << m_iNumChannels;

		std::vector<cv::Mat> channels;
		for (int i = 0; i < m_iNumChannels; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(m_inputGeometry.height, m_inputGeometry.width, CV_32FC1,
				cv::Scalar(mean_value[i]));
			channels.push_back(channel);
		}
		cv::merge(channels, m_mean);
	}
}

void DeepLearning::SetMean( const string& mean_file ) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie( mean_file.c_str(), &blob_proto );

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto( blob_proto );
	if ( mean_blob.channels() != m_iNumChannels )
	{
		std::cout << "Number of channels of mean file doesn't match input layer.\n";
		return;
	}

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for ( int i = 0; i < m_iNumChannels; ++i ) {
		/* Extract an individual channel. */
		cv::Mat channel( mean_blob.height(), mean_blob.width(), CV_32FC1, data );
		channels.push_back( channel );
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge( channels, mean );

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean( mean );
	m_mean = cv::Mat( m_inputGeometry, mean.type(), channel_mean );
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void DeepLearning::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = NET_TYPE(m_network)->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void DeepLearning::WrapInputLayer_Batch( std::vector<std::vector<cv::Mat>>& input_channels_batch )
{
	Blob<float>* input_layer = NET_TYPE( m_network )->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for ( int batch = 0; batch < input_channels_batch.size(); batch++ )
	{
		auto& input_channels = input_channels_batch[batch];
		for ( int i = 0; i < input_layer->channels(); ++i ) {
			cv::Mat channel( height, width, CV_32FC1, input_data );
			input_channels.push_back( channel );
			input_data += width * height;
		}
	}
}

void DeepLearning::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;

	if (img.channels() == 3 && m_iNumChannels == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && m_iNumChannels == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && m_iNumChannels == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && m_iNumChannels == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img.clone();

	cv::Mat sample_resized;
	if (sample.size() != m_inputGeometry)
		cv::resize(sample, sample_resized, m_inputGeometry);
	else
		sample_resized = sample.clone();

	cv::Mat sample_float;
	if (m_iNumChannels == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, m_mean, sample_normalized);

	sample_normalized *= m_fScale;

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);
	//cv::split(sample_scale, *input_channels);

	//std::cout << 5;
	////CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
	//CHECK((float*)(input_channels->at(0).data)
	//	== NET_TYPE(m_network)->input_blobs()[0]->cpu_data())
	//	<< "Input channels are not wrapping the input layer of the network.";
}

bool DeepLearning::NetworkForward( const cv::Mat& img ) {

	Blob<float>* input_layer = NET_TYPE( m_network )->input_blobs()[0];

	input_layer->Reshape( 1, m_iNumChannels,
		m_inputGeometry.height, m_inputGeometry.width);

	NET_TYPE(m_network)->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	NET_TYPE(m_network)->Forward();

	return true;
}

bool DeepLearning::NetworkForward_Batch( const std::vector<cv::Mat>& imgs )
{
	Blob<float>* input_layer = NET_TYPE( m_network )->input_blobs()[0];

	int batchsize = imgs.size();

	input_layer->Reshape( batchsize, m_iNumChannels, m_inputGeometry.height, m_inputGeometry.width);

	NET_TYPE(m_network)->Reshape();

	std::vector<std::vector<cv::Mat>> input_channels_batch(batchsize);

	WrapInputLayer_Batch(input_channels_batch);

	for ( int batch = 0; batch < input_channels_batch.size(); batch++ )
	{
		auto& img = imgs[batch];
		auto& input_channels = input_channels_batch[batch];
		Preprocess(img, &input_channels);
	}

	NET_TYPE(m_network)->Forward();

	return true;
}

bool DeepLearning::Detect_SSD( const cv::Mat& img, std::vector<Detect_Result>& detections ) {
	
	if ( img.empty() )
		return false;

	if ( !m_network )
		return false;

	NetworkForward(img);

	/* Copy the output layer to a std::vector */
	Blob<float>* result_blob = NET_TYPE(m_network)->output_blobs()[0];
	const float* result = result_blob->cpu_data();
	const int num_det = result_blob->height();

	if ( !result_blob )
		return false;
	if ( !result )
		return false;

	detections.clear();
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {
			// Skip invalid detection.
			result += 7;
			continue;
		}

		Detect_Result res;
			
		res.id = static_cast<int>(result[0]);
		res.label = static_cast<int>(result[1]);
		res.score = result[2];
		res.bbox[0] = result[3] * img.cols;
		res.bbox[1] = result[4] * img.rows;
		res.bbox[2] = result[5] * img.cols;
		res.bbox[3] = result[6] * img.rows;

		detections.push_back( res );

		result += 7;
	}

	return true;
}

bool DeepLearning::Detect_SSD_Batch( const std::vector<cv::Mat>& imgs, std::vector<Detect_Result_Batch>& detections_batch )
{
	if ( imgs.size() == 0 )
		return false;

	for ( auto& img : imgs )
	{
		if ( img.empty() )
			return false;
	}

	if ( !m_network )
		return false;

	NetworkForward_Batch( imgs );

	Blob<float>* result_blob = NET_TYPE( m_network )->output_blobs()[0];
	const float* result = result_blob->cpu_data();

	const int num_det = result_blob->height();

	if ( !result_blob )
		return false;
	if ( !result )
		return false;

	detections_batch.clear();

	for ( int k = 0; k < num_det; ++k ) {
		if ( result[0] == -1 ) {
			// Skip invalid detection.
			result += 7;
			continue;
		}

		Detect_Result res;

		res.id = static_cast<int>( result[0] );
		while ( res.id + 1 > detections_batch.size() )
		{
			detections_batch.push_back( Detect_Result_Batch() );
		}

		res.label = static_cast<int>( result[1] );
		res.score = result[2];
		res.bbox[0] = result[3] * imgs[res.id].cols;
		res.bbox[1] = result[4] * imgs[res.id].rows;
		res.bbox[2] = result[5] * imgs[res.id].cols;
		res.bbox[3] = result[6] * imgs[res.id].rows;

		detections_batch[res.id].vDetectRst.push_back( res );

		result += 7;
	}

	return true;
}

bool DeepLearning::Classify( const cv::Mat& img, std::vector<int>& outputs ) {

	if ( img.empty() )
		return false;

	if ( !m_network )
		return false;

	outputs.clear();

	NetworkForward( img );

	outputs.clear();
	/* Copy the output layer to a std::vector */
	for ( auto result_blob : NET_TYPE( m_network )->output_blobs() )
	{
		if ( !result_blob ) 
			return false;

		const float* result = result_blob->cpu_data();

		if ( !result )
			return false;

		for ( int num_i = 0; num_i < 1 /*result_blob->num()*/; num_i++ )
		{
			float max_value = *result;
			int max_index = 0;
			for ( int i = 1; i < result_blob->channels(); i++ )
			{
				float val = *( result + i );
				if ( val > max_value )
				{
					max_value = val;
					max_index = i;
				}
			}
			//outputs[num_i] = ( max_index );
			outputs.push_back( max_index );
		}

	}

	return true;
}

bool DeepLearning::Classify_Batch( const std::vector<cv::Mat>& imgs, std::vector<Classify_Result>& results )
{
	if ( imgs.size() == 0 )
		return false;

	for ( auto& img : imgs )
	{
		if ( img.empty() )
			return false;
	}

	if ( !m_network )
		return false;

	NetworkForward_Batch( imgs );

	results.resize( imgs.size() );

	for ( auto result_blob : NET_TYPE( m_network )->output_blobs() )
	{
		if ( !result_blob )
			return false;

		const float* const blob_data = result_blob->cpu_data();

		if ( !blob_data )
			return false;

		for ( int num_i = 0; num_i < result_blob->num(); num_i++ )
		{
			int offset = result_blob->offset( num_i );
			const float* const result = blob_data + offset;
			float max_value = *result;
			int max_index = 0;
			for ( int i = 1; i < result_blob->channels(); i++ )
			{
				float val = *( result + i );
				if ( val > max_value )
				{
					max_value = val;
					max_index = i;
				}
			}
			//outputs[num_i] = ( max_index );
			results[num_i].vOutput.push_back(max_index);
		}

	}

	return true;
}

const cv::Size& DeepLearning::GetInputSize() const {
	return m_inputGeometry;
}

bool DeepLearning::SetDevice( int gpu_id )
{
	if ( gpu_id < 0 )
		return false;

	Caffe::set_mode( Caffe::GPU );
	Caffe::SetDevice( gpu_id );

	return true;
}

bool DeepLearning::SetScale( float scale )
{
	m_fScale = scale;

	return true;
}

void DeepLearning::SetBlasThreadNum( int thread_num )
{
	openblas_set_num_threads( thread_num );
}

bool DeepLearning::Init_FRCNN(const string& model_file, const string& config_file, const string& weights_file, int gpu_idx /*= -1 */)
{
	if (gpu_idx >= 0)
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpu_idx);
	}
	else 
	{
		Caffe::set_mode(Caffe::CPU);
	}

	API::Set_Config(config_file);
	m_network = new API::Detector();

	FRCNN_NET_TYPE(m_network)->Set_Model(model_file, weights_file);

	return true;
}

bool DeepLearning::Detect_FRCNN(const cv::Mat& img, std::vector<Detect_Result>& detections)
{
	std::vector<caffe::Frcnn::BBox<float> > results;
	FRCNN_NET_TYPE(m_network)->predict(img, results);

	int num_of_res = results.size();
	detections.resize(num_of_res);

	for (int i = 0; i < num_of_res; i++)
	{
		Detect_Result detect_res;
		auto res = results[i];
		detect_res.id = 0;
		detect_res.score = res.confidence;
		detect_res.label = res.id;
		detect_res.bbox[0] = int(res[0]);
		detect_res.bbox[1] = int(res[1]);
		detect_res.bbox[2] = int(res[2]);
		detect_res.bbox[3] = int(res[3]);

		detections[i] = detect_res;
	}

	return true;
}

bool DeepLearning::Release_FRCNN()
{
	if (m_network)
	{
		delete FRCNN_NET_TYPE(m_network);
	}

	m_network = nullptr;

	return true;
}




