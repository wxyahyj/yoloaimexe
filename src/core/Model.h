#ifndef MODEL_H
#define MODEL_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

template<typename T> T vectorProduct(const std::vector<T> &v)
{
	T product = 1;
	for (auto &i : v) {
		if (i > 0) {
			product *= i;
		}
	}
	return product;
}

static void hwc_to_chw(cv::InputArray src, cv::OutputArray dst)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	for (auto &img : channels) {
		img = img.reshape(1, 1);
	}

	cv::hconcat(channels, dst);
}

static void chw_to_hwc_32f(cv::InputArray src, cv::OutputArray dst)
{
	const cv::Mat srcMat = src.getMat();
	const int channels = srcMat.channels();
	const int height = srcMat.rows;
	const int width = srcMat.cols;
	const int dtype = srcMat.type();
	const int channelStride = height * width;

	cv::Mat flatMat = srcMat.reshape(1, 1);

	std::vector<cv::Mat> channelsVec(channels);
	for (int i = 0; i < channels; i++) {
		channelsVec[i] =
			cv::Mat(height, width, CV_MAKE_TYPE(dtype, 1), flatMat.ptr<float>(0) + i * channelStride);
	}

	cv::merge(channelsVec, dst);
}

class Model {
public:
	Model() {};
	virtual ~Model() {};

	virtual void populateInputOutputNames(const std::unique_ptr<Ort::Session> &session,
					      std::vector<Ort::AllocatedStringPtr> &inputNames,
					      std::vector<Ort::AllocatedStringPtr> &outputNames)
	{
		Ort::AllocatorWithDefaultOptions allocator;

		inputNames.clear();
		outputNames.clear();
		inputNames.push_back(session->GetInputNameAllocated(0, allocator));
		outputNames.push_back(session->GetOutputNameAllocated(0, allocator));
	}

	virtual bool populateInputOutputShapes(const std::unique_ptr<Ort::Session> &session,
					       std::vector<std::vector<int64_t>> &inputDims,
					       std::vector<std::vector<int64_t>> &outputDims)
	{
		inputDims.clear();
		outputDims.clear();

		inputDims.push_back(std::vector<int64_t>());
		outputDims.push_back(std::vector<int64_t>());

		const Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
		const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		outputDims[0] = outputTensorInfo.GetShape();

		for (auto &i : outputDims[0]) {
			if (i == -1) {
				i = 1;
			}
		}

		const Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
		const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		inputDims[0] = inputTensorInfo.GetShape();

		for (auto &i : inputDims[0]) {
			if (i == -1) {
				i = 1;
			}
		}

		if (inputDims[0].size() < 3 || outputDims[0].size() < 3) {
			std::cerr << "Input or output tensor dims are < 3" << std::endl;
			return false;
		}

		return true;
	}

	virtual void allocateTensorBuffers(const std::vector<std::vector<int64_t>> &inputDims,
					   const std::vector<std::vector<int64_t>> &outputDims,
					   std::vector<std::vector<float>> &outputTensorValues,
					   std::vector<std::vector<float>> &inputTensorValues,
					   std::vector<Ort::Value> &inputTensor, std::vector<Ort::Value> &outputTensor)
	{
		outputTensorValues.clear();
		outputTensor.clear();

		inputTensorValues.clear();
		inputTensor.clear();

		Ort::MemoryInfo memoryInfo =
			Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

		for (size_t i = 0; i < inputDims.size(); i++) {
			inputTensorValues.push_back(std::vector<float>(vectorProduct(inputDims[i]), 0.0f));
			inputTensor.push_back(Ort::Value::CreateTensor<float>(
				memoryInfo, inputTensorValues[i].data(), inputTensorValues[i].size(),
				inputDims[i].data(), inputDims[i].size()));
		}

		for (size_t i = 0; i < outputDims.size(); i++) {
			outputTensorValues.push_back(std::vector<float>(vectorProduct(outputDims[i]), 0.0f));
			outputTensor.push_back(Ort::Value::CreateTensor<float>(
				memoryInfo, outputTensorValues[i].data(), outputTensorValues[i].size(),
				outputDims[i].data(), outputDims[i].size()));
		}
	}

	virtual void getNetworkInputSize(const std::vector<std::vector<int64_t>> &inputDims, uint32_t &inputWidth,
					 uint32_t &inputHeight)
	{
		inputWidth = (int)inputDims[0][2];
		inputHeight = (int)inputDims[0][1];
	}

	virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
	{
		preprocessedImage = resizedImage / 255.0;
	}

	virtual void postprocessOutput(cv::Mat &output) {}

	virtual void loadInputToTensor(const cv::Mat &preprocessedImage, uint32_t inputWidth, uint32_t inputHeight,
				       std::vector<std::vector<float>> &inputTensorValues)
	{
		preprocessedImage.copyTo(cv::Mat(inputHeight, inputWidth, CV_32FC3, &(inputTensorValues[0][0])));
	}

	virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims,
					 std::vector<std::vector<float>> &outputTensorValues)
	{
		uint32_t outputWidth = (int)outputDims[0].at(2);
		uint32_t outputHeight = (int)outputDims[0].at(1);
		int32_t outputChannels = CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(3));

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

	virtual void assignOutputToInput(std::vector<std::vector<float>> &, std::vector<std::vector<float>> &) {}

	virtual void runNetworkInference(const std::unique_ptr<Ort::Session> &session,
					 const std::vector<Ort::AllocatedStringPtr> &inputNames,
					 const std::vector<Ort::AllocatedStringPtr> &outputNames,
					 const std::vector<Ort::Value> &inputTensor,
					 std::vector<Ort::Value> &outputTensor)
	{
		if (inputNames.size() == 0 || outputNames.size() == 0 || inputTensor.size() == 0 ||
		    outputTensor.size() == 0) {
			std::cout << "Skip network inference. Inputs or outputs are null." << std::endl;
			return;
		}

		std::vector<const char *> rawInputNames;
		for (auto &inputName : inputNames) {
			rawInputNames.push_back(inputName.get());
		}

		std::vector<const char *> rawOutputNames;
		for (auto &outputName : outputNames) {
			rawOutputNames.push_back(outputName.get());
		}

		session->Run(Ort::RunOptions{nullptr}, rawInputNames.data(), inputTensor.data(), inputNames.size(),
			     rawOutputNames.data(), outputTensor.data(), outputNames.size());
	}
};

class ModelBCHW : public Model {
public:
	ModelBCHW() {}
	~ModelBCHW() {}

	virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
	{
		resizedImage = resizedImage / 255.0;
		hwc_to_chw(resizedImage, preprocessedImage);
	}

	virtual void postprocessOutput(cv::Mat &outputImage)
	{
		cv::Mat outputTransposed;
		chw_to_hwc_32f(outputImage, outputTransposed);
		std::vector<cv::Mat> outputImageSplit;
		cv::split(outputTransposed, outputImageSplit);
		outputImage = outputImageSplit[1];
	}

	virtual void getNetworkInputSize(const std::vector<std::vector<int64_t>> &inputDims, uint32_t &inputWidth,
					 uint32_t &inputHeight)
	{
		inputWidth = (int)inputDims[0][3];
		inputHeight = (int)inputDims[0][2];
	}

	virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims,
					 std::vector<std::vector<float>> &outputTensorValues)
	{
		uint32_t outputWidth = (int)outputDims[0].at(3);
		uint32_t outputHeight = (int)outputDims[0].at(2);
		int32_t outputChannels = CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(1));

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

	virtual void loadInputToTensor(const cv::Mat &preprocessedImage, uint32_t, uint32_t,
				       std::vector<std::vector<float>> &inputTensorValues)
	{
		inputTensorValues[0].assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());
	}
};

#endif
