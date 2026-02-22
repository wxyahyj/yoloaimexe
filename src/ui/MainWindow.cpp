#include "MainWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , isRunning_(false)
    , modelLoaded_(false)
{
    setupUI();
    loadModel();
}

MainWindow::~MainWindow() {
    if (timer_) {
        timer_->stop();
    }
    if (cap_ && cap_->isOpened()) {
        cap_->release();
    }
}

void MainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    
    videoLabel_ = new QLabel(this);
    videoLabel_->setMinimumSize(800, 600);
    videoLabel_->setAlignment(Qt::AlignCenter);
    videoLabel_->setText("Waiting for camera...");
    mainLayout->addWidget(videoLabel_);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    
    startStopBtn_ = new QPushButton("Start", this);
    startStopBtn_->setEnabled(false);
    connect(startStopBtn_, &QPushButton::clicked, this, &MainWindow::onStartStopClicked);
    buttonLayout->addWidget(startStopBtn_);
    
    settingsBtn_ = new QPushButton("Settings", this);
    connect(settingsBtn_, &QPushButton::clicked, this, &MainWindow::onSettingsClicked);
    buttonLayout->addWidget(settingsBtn_);
    
    mainLayout->addLayout(buttonLayout);
    
    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &MainWindow::updateFrame);
    
    settingsPanel_ = new SettingsPanel(this);
    connect(settingsPanel_, &SettingsPanel::settingsChanged, this, &MainWindow::onSettingsChanged);
    
    setWindowTitle("YOLO Detector");
    resize(900, 700);
}

void MainWindow::loadModel() {
    try {
        model_ = std::make_unique<ModelYOLO>(ModelYOLO::Version::YOLOv8);
        model_->loadModel("models/yolov8n.onnx", "cpu", 4, 640);
        model_->setConfidenceThreshold(0.5f);
        model_->setNMSThreshold(0.45f);
        modelLoaded_ = true;
        startStopBtn_->setEnabled(true);
        QMessageBox::information(this, "Success", "Model loaded successfully!");
    } catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", 
            QString("Failed to load model: %1\nPlease put your .onnx model in the models/ folder.").arg(e.what()));
    }
}

void MainWindow::onStartStopClicked() {
    if (!isRunning_) {
        cap_ = std::make_unique<cv::VideoCapture>(0);
        if (!cap_->isOpened()) {
            QMessageBox::warning(this, "Error", "Failed to open camera!");
            return;
        }
        isRunning_ = true;
        startStopBtn_->setText("Stop");
        timer_->start(30);
    } else {
        timer_->stop();
        if (cap_) {
            cap_->release();
        }
        isRunning_ = false;
        startStopBtn_->setText("Start");
        videoLabel_->setText("Camera stopped");
    }
}

void MainWindow::onSettingsClicked() {
    settingsPanel_->show();
}

void MainWindow::updateFrame() {
    if (!cap_ || !modelLoaded_) return;
    
    cv::Mat frame;
    *cap_ >> frame;
    if (frame.empty()) return;
    
    cv::Mat displayFrame = frame.clone();
    
    auto detections = model_->inference(frame);
    drawDetections(displayFrame, detections);
    
    cv::Mat rgbFrame = convertToRGB(displayFrame);
    QImage qimg(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, 
                rgbFrame.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qimg);
    videoLabel_->setPixmap(pixmap.scaled(videoLabel_->size(), 
                                          Qt::KeepAspectRatio, 
                                          Qt::SmoothTransformation));
}

void MainWindow::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        int x = static_cast<int>(det.x * frame.cols);
        int y = static_cast<int>(det.y * frame.rows);
        int w = static_cast<int>(det.width * frame.cols);
        int h = static_cast<int>(det.height * frame.rows);
        
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h),
                     cv::Scalar(0, 255, 0), 2);
        
        std::string label = det.className + ": " + 
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        cv::putText(frame, label, cv::Point(x, y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}

cv::Mat MainWindow::convertToRGB(const cv::Mat& frame) {
    cv::Mat rgb;
    if (frame.channels() == 4) {
        cv::cvtColor(frame, rgb, cv::COLOR_BGRA2RGB);
    } else if (frame.channels() == 3) {
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = frame.clone();
    }
    return rgb;
}

void MainWindow::onModelLoaded() {
    modelLoaded_ = true;
    startStopBtn_->setEnabled(true);
}

void MainWindow::onSettingsChanged() {
    if (model_) {
        model_->setConfidenceThreshold(settingsPanel_->getConfidenceThreshold());
        model_->setNMSThreshold(settingsPanel_->getNMSThreshold());
    }
}
