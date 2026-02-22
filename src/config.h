#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace Config {
    // 默认模型路径
    inline const std::string DEFAULT_MODEL_PATH = "models/yolov8n.onnx";
    
    // 默认设备
    inline const std::string DEFAULT_DEVICE = "cpu";
    
    // 默认线程数
    inline const int DEFAULT_NUM_THREADS = 4;
    
    // 默认置信度阈值
    inline const float DEFAULT_CONFIDENCE_THRESHOLD = 0.5f;
    
    // 默认NMS阈值
    inline const float DEFAULT_NMS_THRESHOLD = 0.45f;
    
    // 默认输入分辨率
    inline const int DEFAULT_INPUT_RESOLUTION = 640;
    
    // 默认摄像头ID
    inline const int DEFAULT_CAMERA_ID = 0;
}

#endif
