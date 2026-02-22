# YOLO Detector EXE

一个独立的YOLO目标检测应用程序，具有图形用户界面，支持实时摄像头检测。

## 功能特性

- 实时摄像头YOLO目标检测
- 图形用户界面（基于Qt）
- 可调节的检测参数：
  - 置信度阈值
  - NMS阈值
  - 线程数
  - 计算设备（CPU/CUDA/DirectML）
- GitHub Actions云端自动构建

## 快速开始

### 使用GitHub Actions构建（推荐）

1. 将此项目推送到GitHub仓库
2. 进入GitHub仓库的 **Actions** 标签页
3. 选择 **Build YOLO Detector** 工作流
4. 点击 **Run workflow** 手动触发构建
5. 构建完成后，在 **Artifacts** 中下载编译好的程序

### 本地构建

#### 依赖要求

- CMake 3.16+
- Qt 6.5+
- OpenCV 4.x
- ONNX Runtime 1.15+
- C++17 编译器

#### Windows构建步骤

```bash
# 1. 克隆项目
cd 重构exe

# 2. 下载ONNX Runtime
# 从 https://github.com/microsoft/onnxruntime/releases 下载
# 解压到 onnxruntime/ 目录

# 3. 配置CMake
cmake -B build -S . -DCMAKE_PREFIX_PATH="path/to/Qt6"

# 4. 编译
cmake --build build --config Release

# 5. 运行
build/bin/Release/yolo_detector_exe.exe
```

## 使用说明

1. **放置模型文件**：将你的YOLO ONNX模型文件（如 `yolov8n.onnx`）放入 `models/` 文件夹
2. **启动程序**：运行 `yolo_detector_exe.exe`
3. **开始检测**：点击 **Start** 按钮开始摄像头检测
4. **调整参数**：点击 **Settings** 按钮打开设置面板

## 设置参数

### 检测设置

- **Confidence Threshold**（置信度阈值）：0.0 - 1.0，越高越严格
- **NMS Threshold**（非极大值抑制阈值）：0.0 - 1.0，用于去除重叠框

### 性能设置

- **Number of Threads**（线程数）：1 - 16
- **Device**（计算设备）：
  - CPU
  - CUDA（需要NVIDIA显卡和CUDA支持）
  - DirectML（Windows上的GPU加速）

## 项目结构

```
重构exe/
├── .github/workflows/
│   └── build.yml          # GitHub Actions构建配置
├── src/
│   ├── main.cpp            # 程序入口
│   ├── config.h            # 配置文件
│   ├── core/               # 核心算法模块
│   │   ├── Model.h         # 模型基类
│   │   ├── ModelYOLO.h     # YOLO模型头文件
│   │   ├── ModelYOLO.cpp   # YOLO模型实现
│   │   └── Detection.h     # 检测结果定义
│   └── ui/                 # UI模块
│       ├── MainWindow.h    # 主窗口
│       ├── MainWindow.cpp
│       ├── SettingsPanel.h # 设置面板
│       └── SettingsPanel.cpp
├── models/                 # 模型文件夹（放.onnx文件）
├── CMakeLists.txt          # CMake构建配置
├── .gitignore
└── README.md
```

## 技术栈

- **GUI框架**：Qt 6
- **图像处理**：OpenCV
- **推理引擎**：ONNX Runtime
- **构建系统**：CMake
- **CI/CD**：GitHub Actions

## 许可证

基于原OBS Background Removal项目重构。

## 致谢

- YOLO目标检测算法
- ONNX Runtime
- Qt框架
- OpenCV
