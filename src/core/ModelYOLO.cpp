#include "ModelYOLO.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <iomanip>
#ifdef HAVE_ONNXRUNTIME_DML_EP
#include <dml_provider_factory.h>
#endif

ModelYOLO::ModelYOLO(Version version)
    : ModelBCHW(),
      version_(version),
      confidenceThreshold_(0.5f),
      nmsThreshold_(0.45f),
      targetClassId_(-1),
      inputWidth_(640),
      inputHeight_(640),
      numClasses_(80),
      inputBufferSize_(0)
{
    std::cout << "[ModelYOLO] Initialized (Version: " << static_cast<int>(version) << ")" << std::endl;
    
    try {
        std::string instanceName{"YOLOModel"};
        env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str());
    } catch (const std::exception& e) {
        std::cerr << "[ModelYOLO] Failed to initialize ORT: " << e.what() << std::endl;
    }
}

ModelYOLO::~ModelYOLO() {
    std::cout << "[ModelYOLO] Destroyed" << std::endl;
}

void ModelYOLO::loadModel(const std::string& modelPath, const std::string& useGPU, int numThreads, int inputResolution) {
    std::cout << "[ModelYOLO] Loading model: " << modelPath << std::endl;
    
    std::string currentUseGPU = useGPU;
    bool gpuFailed = false;
    
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        std::cout << "[ModelYOLO] Using device: " << currentUseGPU << std::endl;
        
        if (currentUseGPU != "cpu") {
            sessionOptions.DisableMemPattern();
            sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        } else {
            sessionOptions.SetInterOpNumThreads(numThreads);
            sessionOptions.SetIntraOpNumThreads(numThreads);
        }
        
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
        if (currentUseGPU == "cuda") {
            std::cout << "[ModelYOLO] Attempting to enable CUDA execution provider..." << std::endl;
            try {
                std::cout << "[ModelYOLO] Loading CUDA execution provider with device ID 0" << std::endl;
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
                std::cout << "[ModelYOLO] CUDA execution provider enabled successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ModelYOLO] Failed to enable CUDA: " << e.what() << ", falling back to CPU" << std::endl;
                std::cout << "[ModelYOLO] CUDA execution provider fallback to CPU mode" << std::endl;
                std::cout << "[ModelYOLO] Possible reasons: missing cuDNN, incorrect CUDA version, or missing dependencies" << std::endl;
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif
#ifdef HAVE_ONNXRUNTIME_ROCM_EP
        if (currentUseGPU == "rocm" && !gpuFailed) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(sessionOptions, 0));
                std::cout << "[ModelYOLO] ROCM execution provider enabled" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ModelYOLO] Failed to enable ROCM: " << e.what() << ", falling back to CPU" << std::endl;
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif
#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
        if (currentUseGPU == "tensorrt" && !gpuFailed) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
                std::cout << "[ModelYOLO] TensorRT execution provider enabled" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ModelYOLO] Failed to enable TensorRT: " << e.what() << ", falling back to CPU" << std::endl;
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif

#ifdef HAVE_ONNXRUNTIME_DML_EP
        if (currentUseGPU == "dml" && !gpuFailed) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));
                std::cout << "[ModelYOLO] DirectML execution provider enabled" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ModelYOLO] Failed to enable DirectML: " << e.what() << ", falling back to CPU" << std::endl;
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif
        
        if (gpuFailed) {
            sessionOptions.SetInterOpNumThreads(numThreads);
            sessionOptions.SetIntraOpNumThreads(numThreads);
            std::cout << "[ModelYOLO] Switched to CPU mode" << std::endl;
        }
        
#if _WIN32
        std::wstring modelPathW(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(*env_, modelPathW.c_str(), sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
#endif
        
        populateInputOutputNames(session_, inputNames_, outputNames_);
        populateInputOutputShapes(session_, inputDims_, outputDims_);
        
        if (!inputDims_.empty()) {
            auto shape = inputDims_[0];
            if (shape.size() >= 4) {
                inputHeight_ = static_cast<int>(shape[2]);
                inputWidth_ = static_cast<int>(shape[3]);
                std::cout << "[ModelYOLO] Using model actual input size: " << inputWidth_ << "x" << inputHeight_ << std::endl;
            }
        }
        
        allocateTensorBuffers(inputDims_, outputDims_, outputTensorValues_, inputTensorValues_,
                              inputTensor_, outputTensor_);
        
        if (!outputDims_.empty()) {
            auto shape = outputDims_[0];
            std::cout << "[ModelYOLO] Output shape size: " << shape.size() << std::endl;
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << "[ModelYOLO] Output shape[" << i << "]: " << shape[i] << std::endl;
            }
            std::cout << "[ModelYOLO] Model version: " << static_cast<int>(version_) << std::endl;
            
            int detectedClasses = 80;
            
            if (version_ == Version::YOLOv5 && shape.size() >= 3) {
                int64_t lastDim = shape[2];
                if (lastDim > 5) {
                    detectedClasses = static_cast<int>(lastDim - 5);
                }
                std::cout << "[ModelYOLO] YOLOv5 mode: lastDim=" << lastDim << ", detectedClasses=" << detectedClasses << std::endl;
            } else if (shape.size() >= 3) {
                int64_t elementsDim = shape[1];
                if (elementsDim > 4) {
                    detectedClasses = static_cast<int>(elementsDim - 4);
                }
                std::cout << "[ModelYOLO] YOLOv8/v11 mode: elementsDim=" << elementsDim << ", detectedClasses=" << detectedClasses << std::endl;
            }
            
            if (detectedClasses > 0 && detectedClasses < 1000) {
                numClasses_ = detectedClasses;
                std::cout << "[ModelYOLO] Using numClasses: " << numClasses_ << " (valid range)" << std::endl;
            } else {
                std::cerr << "[ModelYOLO] Detected numClasses " << detectedClasses << " is invalid, using default: 80" << std::endl;
                numClasses_ = 80;
            }
        }
        
        inputBufferSize_ = 1 * 3 * inputHeight_ * inputWidth_;
        inputBuffer_.resize(inputBufferSize_);
        std::cout << "[ModelYOLO] Allocated input buffer size: " << inputBufferSize_ << std::endl;
        
        name = "YOLO";
        
        std::cout << "[ModelYOLO] Model loaded successfully" << std::endl;
        std::cout << "  Input size: " << inputWidth_ << "x" << inputHeight_ << std::endl;
        std::cout << "  Num classes: " << numClasses_ << std::endl;
        std::cout << "  Device: " << currentUseGPU << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ModelYOLO] Failed to load model: " << e.what() << std::endl;
        throw;
    }
}

void ModelYOLO::preprocessInput(const cv::Mat& input, float* outputBuffer) {
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(inputWidth_, inputHeight_));

    cv::Mat rgb;
    if (input.channels() == 4) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
    } else if (input.channels() == 3) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = resized.clone();
    }

    cv::Mat rgb8u;
    if (rgb.depth() != CV_8U) {
        rgb.convertTo(rgb8u, CV_8U);
    } else {
        rgb8u = rgb;
    }

    const int channelSize = inputWidth_ * inputHeight_;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < inputHeight_; ++h) {
            for (int w = 0; w < inputWidth_; ++w) {
                int outputIdx = c * channelSize + h * inputWidth_ + w;
                outputBuffer[outputIdx] = rgb8u.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}

std::vector<Detection> ModelYOLO::inference(const cv::Mat& input) {
    
    if (input.empty()) {
        std::cerr << "[ModelYOLO] Input image is empty" << std::endl;
        return {};
    }
    
    if (input.cols <= 0 || input.rows <= 0) {
        std::cerr << "[ModelYOLO] Invalid input image size: " << input.cols << "x" << input.rows << std::endl;
        return {};
    }
    
    if (!session_) {
        std::cerr << "[ModelYOLO] Session is null, cannot run inference" << std::endl;
        return {};
    }
    
    try {
        preprocessInput(input, inputBuffer_.data());
        
        std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
        
        Ort::Value inputTensor;
        try {
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, 
                OrtMemType::OrtMemTypeDefault
            );
            
            inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputBuffer_.data(),
                inputBufferSize_,
                inputShape.data(),
                inputShape.size()
            );
        } catch (const std::exception& e) {
            std::cerr << "[ModelYOLO] Failed to create input tensor: " << e.what() << std::endl;
            return {};
        }
        
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(inputTensor));
        
        std::vector<const char*> inputNamesChar;
        for (const auto& name : inputNames_) {
            inputNamesChar.push_back(name.get());
        }
        
        std::vector<const char*> outputNamesChar;
        for (const auto& name : outputNames_) {
            outputNamesChar.push_back(name.get());
        }

        Ort::RunOptions runOptions;
        
        std::vector<Ort::Value> outputTensors;
        try {
            outputTensors = session_->Run(
                runOptions,
                inputNamesChar.data(),
                inputTensors.data(),
                inputTensors.size(),
                outputNamesChar.data(),
                outputNamesChar.size()
            );
        } catch (const Ort::Exception& e) {
            std::cerr << "[ModelYOLO] ONNX Runtime exception during Run: " << e.what() << std::endl;
            return {};
        } catch (const std::exception& e) {
            std::cerr << "[ModelYOLO] Exception during Run: " << e.what() << std::endl;
            return {};
        } catch (...) {
            std::cerr << "[ModelYOLO] Unknown exception during Run" << std::endl;
            return {};
        }
        
        if (outputTensors.empty()) {
            std::cerr << "[ModelYOLO] No output tensors from ONNX Runtime" << std::endl;
            return {};
        }
        
        if (!outputTensors[0].IsTensor()) {
            std::cerr << "[ModelYOLO] Output is not a tensor" << std::endl;
            return {};
        }
        
        float* outputData = nullptr;
        try {
            outputData = outputTensors[0].GetTensorMutableData<float>();
        } catch (const std::exception& e) {
            std::cerr << "[ModelYOLO] Failed to get output tensor data: " << e.what() << std::endl;
            return {};
        }
        
        if (!outputData) {
            std::cerr << "[ModelYOLO] Failed to get output tensor data" << std::endl;
            return {};
        }
        
        std::vector<int64_t> outputShape;
        try {
            outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        } catch (const std::exception& e) {
            std::cerr << "[ModelYOLO] Failed to get output shape: " << e.what() << std::endl;
            return {};
        }
        
        if (outputShape.size() < 3) {
            std::cerr << "[ModelYOLO] Invalid output shape size: " << outputShape.size() << std::endl;
            return {};
        }
        
        int numBoxes = 0, numElements = 0;
        
        try {
            if (version_ == Version::YOLOv5) {
                numBoxes = static_cast<int>(outputShape[1]);
                numElements = static_cast<int>(outputShape[2]);
            } else {
                numBoxes = static_cast<int>(outputShape[2]);
                numElements = static_cast<int>(outputShape[1]);
            }
        } catch (const std::exception& e) {
            std::cerr << "[ModelYOLO] Failed to parse output shape: " << e.what() << std::endl;
            return {};
        }
        
        if (numBoxes <= 0 || numElements <= 0) {
            std::cerr << "[ModelYOLO] Invalid output parameters: numBoxes=" << numBoxes << ", numElements=" << numElements << std::endl;
            return {};
        }

        cv::Size modelSize(inputWidth_, inputHeight_);
        cv::Size originalSize(input.cols, input.rows);
        
        std::vector<Detection> detections;
        
        try {
            switch (version_) {
                case Version::YOLOv5:
                    detections = postprocessYOLOv5(outputData, numBoxes, numClasses_, 
                                                  modelSize, originalSize);
                    break;
                case Version::YOLOv8:
                    detections = postprocessYOLOv8(outputData, numBoxes, numClasses_, 
                                                  modelSize, originalSize);
                    break;
                case Version::YOLOv11:
                    detections = postprocessYOLOv11(outputData, numBoxes, numClasses_, 
                                                   modelSize, originalSize);
                    break;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ModelYOLO] Postprocessing exception: " << e.what() << std::endl;
            return {};
        }
        
        return detections;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[ModelYOLO] ONNX Runtime exception: " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cerr << "[ModelYOLO] Inference exception: " << e.what() << std::endl;
        return {};
    } catch (...) {
        std::cerr << "[ModelYOLO] Unknown inference exception" << std::endl;
        return {};
    }
}

std::vector<Detection> ModelYOLO::postprocessYOLOv5(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    const int numElements = 5 + numClasses;

    float scaleX = static_cast<float>(originalImageSize.width) / modelInputSize.width;
    float scaleY = static_cast<float>(originalImageSize.height) / modelInputSize.height;

    for (int i = 0; i < numBoxes; ++i) {
        const float* detection = rawOutput + i * numElements;

        float objectness = detection[4];

        if (objectness < confidenceThreshold_) {
            continue;
        }

        int maxClassId = 0;
        float maxClassProb = detection[5];

        for (int c = 1; c < numClasses; ++c) {
            if (detection[5 + c] > maxClassProb) {
                maxClassProb = detection[5 + c];
                maxClassId = c;
            }
        }

        float confidence = objectness * maxClassProb;

        if (confidence < confidenceThreshold_) {
            continue;
        }

        bool isTargetClass = false;
        if (targetClassId_ >= 0) {
            isTargetClass = (maxClassId == targetClassId_);
        } else if (!targetClasses_.empty()) {
            isTargetClass = (std::find(targetClasses_.begin(), targetClasses_.end(), maxClassId) != targetClasses_.end());
        } else {
            isTargetClass = true;
        }
        
        if (!isTargetClass) {
            continue;
        }

        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];

        float x1 = (cx - w / 2.0f) * scaleX;
        float y1 = (cy - h / 2.0f) * scaleY;
        float x2 = (cx + w / 2.0f) * scaleX;
        float y2 = (cy + h / 2.0f) * scaleY;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(originalImageSize.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(originalImageSize.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(originalImageSize.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(originalImageSize.height)));

        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(confidence);
        classIds.push_back(maxClassId);
    }

    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);

    for (int idx : nmsIndices) {
        Detection det;
        det.classId = classIds[idx];
        det.className = (det.classId < classNames_.size())
                        ? classNames_[det.classId]
                        : "Class_" + std::to_string(det.classId);
        det.confidence = scores[idx];

        det.x = boxes[idx].x / originalImageSize.width;
        det.y = boxes[idx].y / originalImageSize.height;
        det.width = boxes[idx].width / originalImageSize.width;
        det.height = boxes[idx].height / originalImageSize.height;

        det.centerX = det.x + det.width / 2.0f;
        det.centerY = det.y + det.height / 2.0f;

        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> ModelYOLO::postprocessYOLOv8(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    float scaleX = static_cast<float>(originalImageSize.width) / modelInputSize.width;
    float scaleY = static_cast<float>(originalImageSize.height) / modelInputSize.height;

    for (int i = 0; i < numBoxes; ++i) {
        float cx = rawOutput[0 * numBoxes + i];
        float cy = rawOutput[1 * numBoxes + i];
        float w = rawOutput[2 * numBoxes + i];
        float h = rawOutput[3 * numBoxes + i];

        int maxClassId = 0;
        float maxClassProb = rawOutput[4 * numBoxes + i];

        for (int c = 1; c < numClasses; ++c) {
            float prob = rawOutput[(4 + c) * numBoxes + i];
            if (prob > maxClassProb) {
                maxClassProb = prob;
                maxClassId = c;
            }
        }

        float confidence = maxClassProb;

        if (confidence < confidenceThreshold_) {
            continue;
        }

        bool isTargetClass = false;
        if (targetClassId_ >= 0) {
            isTargetClass = (maxClassId == targetClassId_);
        } else if (!targetClasses_.empty()) {
            isTargetClass = (std::find(targetClasses_.begin(), targetClasses_.end(), maxClassId) != targetClasses_.end());
        } else {
            isTargetClass = true;
        }
        
        if (!isTargetClass) {
            continue;
        }

        float x1 = (cx - w / 2.0f) * scaleX;
        float y1 = (cy - h / 2.0f) * scaleY;
        float x2 = (cx + w / 2.0f) * scaleX;
        float y2 = (cy + h / 2.0f) * scaleY;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(originalImageSize.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(originalImageSize.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(originalImageSize.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(originalImageSize.height)));

        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(confidence);
        classIds.push_back(maxClassId);
    }

    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);

    for (int idx : nmsIndices) {
        Detection det;
        det.classId = classIds[idx];
        det.className = (det.classId < classNames_.size())
                        ? classNames_[det.classId]
                        : "Class_" + std::to_string(det.classId);
        det.confidence = scores[idx];

        det.x = boxes[idx].x / originalImageSize.width;
        det.y = boxes[idx].y / originalImageSize.height;
        det.width = boxes[idx].width / originalImageSize.width;
        det.height = boxes[idx].height / originalImageSize.height;

        det.centerX = det.x + det.width / 2.0f;
        det.centerY = det.y + det.height / 2.0f;

        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> ModelYOLO::postprocessYOLOv11(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    return postprocessYOLOv8(rawOutput, numBoxes, numClasses, modelInputSize, originalImageSize);
}

std::vector<int> ModelYOLO::performNMS(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float nmsThreshold
) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];

        if (suppressed[idx]) {
            continue;
        }

        keep.push_back(idx);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];

            if (suppressed[idx2]) {
                continue;
            }

            float iou = calculateIoU(boxes[idx], boxes[idx2]);

            if (iou > nmsThreshold) {
                suppressed[idx2] = true;
            }
        }
    }

    return keep;
}

float ModelYOLO::calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    if (x2 < x1 || y2 < y1) {
        return 0.0f;
    }

    float intersection = (x2 - x1) * (y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float unionArea = areaA + areaB - intersection;

    return intersection / unionArea;
}

void ModelYOLO::xywhToxyxy(float cx, float cy, float w, float h,
                            float& x1, float& y1, float& x2, float& y2) {
    x1 = cx - w / 2.0f;
    y1 = cy - h / 2.0f;
    x2 = cx + w / 2.0f;
    y2 = cy + h / 2.0f;
}

void ModelYOLO::loadClassNames(const std::string& namesFile) {
    std::ifstream file(namesFile);

    if (!file.is_open()) {
        std::cerr << "[ModelYOLO] Failed to open class names: " << namesFile << std::endl;
        return;
    }

    classNames_.clear();
    std::string line;

    while (std::getline(file, line)) {
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        if (!line.empty()) {
            classNames_.push_back(line);
        }
    }

    numClasses_ = static_cast<int>(classNames_.size());

    std::cout << "[ModelYOLO] Loaded " << numClasses_ << " class names" << std::endl;
}

void ModelYOLO::setConfidenceThreshold(float threshold) {
    confidenceThreshold_ = std::max(0.0f, std::min(threshold, 1.0f));
}

void ModelYOLO::setNMSThreshold(float threshold) {
    nmsThreshold_ = std::max(0.0f, std::min(threshold, 1.0f));
}

void ModelYOLO::setTargetClass(int classId) {
    targetClassId_ = classId;
    targetClasses_.clear();
    if (classId >= 0) {
        targetClasses_.push_back(classId);
    }
}

void ModelYOLO::setTargetClasses(const std::vector<int>& classIds) {
    targetClasses_ = classIds;
    if (classIds.size() == 1) {
        targetClassId_ = classIds[0];
    } else if (classIds.empty()) {
        targetClassId_ = -1;
    } else {
        targetClassId_ = -1;
    }
}

void ModelYOLO::setInputResolution(int resolution) {
    std::cout << "[ModelYOLO] setInputResolution is disabled. Input resolution is determined by model." << std::endl;
    std::cout << "[ModelYOLO] Current model input size: " << inputWidth_ << "x" << inputHeight_ << std::endl;
}
