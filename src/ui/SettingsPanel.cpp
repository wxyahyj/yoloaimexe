#include "SettingsPanel.h"
#include <QFormLayout>

SettingsPanel::SettingsPanel(QWidget *parent)
    : QWidget(parent)
    , confidenceThreshold_(0.5f)
    , nmsThreshold_(0.45f)
{
    setupUI();
    setWindowTitle("Settings");
    resize(400, 350);
}

void SettingsPanel::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    QGroupBox* detectionGroup = new QGroupBox("Detection Settings", this);
    QFormLayout* detectionLayout = new QFormLayout(detectionGroup);
    
    confidenceSlider_ = new QSlider(Qt::Horizontal, this);
    confidenceSlider_->setRange(0, 100);
    confidenceSlider_->setValue(50);
    confidenceLabel_ = new QLabel("0.50", this);
    connect(confidenceSlider_, &QSlider::valueChanged, this, &SettingsPanel::onConfidenceChanged);
    
    QHBoxLayout* confidenceLayout = new QHBoxLayout();
    confidenceLayout->addWidget(confidenceSlider_);
    confidenceLayout->addWidget(confidenceLabel_);
    
    detectionLayout->addRow("Confidence Threshold:", confidenceLayout);
    
    nmsSlider_ = new QSlider(Qt::Horizontal, this);
    nmsSlider_->setRange(0, 100);
    nmsSlider_->setValue(45);
    nmsLabel_ = new QLabel("0.45", this);
    connect(nmsSlider_, &QSlider::valueChanged, this, &SettingsPanel::onNMSChanged);
    
    QHBoxLayout* nmsLayout = new QHBoxLayout();
    nmsLayout->addWidget(nmsSlider_);
    nmsLayout->addWidget(nmsLabel_);
    
    detectionLayout->addRow("NMS Threshold:", nmsLayout);
    
    mainLayout->addWidget(detectionGroup);
    
    QGroupBox* performanceGroup = new QGroupBox("Performance Settings", this);
    QFormLayout* performanceLayout = new QFormLayout(performanceGroup);
    
    threadsSpinBox_ = new QSpinBox(this);
    threadsSpinBox_->setRange(1, 16);
    threadsSpinBox_->setValue(4);
    performanceLayout->addRow("Number of Threads:", threadsSpinBox_);
    
    deviceCombo_ = new QComboBox(this);
    deviceCombo_->addItem("CPU", "cpu");
    deviceCombo_->addItem("CUDA", "cuda");
    deviceCombo_->addItem("DirectML", "dml");
    performanceLayout->addRow("Device:", deviceCombo_);
    
    mainLayout->addWidget(performanceGroup);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    
    applyBtn_ = new QPushButton("Apply", this);
    connect(applyBtn_, &QPushButton::clicked, this, &SettingsPanel::onApplyClicked);
    buttonLayout->addWidget(applyBtn_);
    
    resetBtn_ = new QPushButton("Reset", this);
    connect(resetBtn_, &QPushButton::clicked, this, &SettingsPanel::onResetClicked);
    buttonLayout->addWidget(resetBtn_);
    
    mainLayout->addLayout(buttonLayout);
    mainLayout->addStretch();
}

void SettingsPanel::onConfidenceChanged(int value) {
    confidenceThreshold_ = value / 100.0f;
    confidenceLabel_->setText(QString::number(confidenceThreshold_, 'f', 2));
}

void SettingsPanel::onNMSChanged(int value) {
    nmsThreshold_ = value / 100.0f;
    nmsLabel_->setText(QString::number(nmsThreshold_, 'f', 2));
}

void SettingsPanel::onApplyClicked() {
    emit settingsChanged();
}

void SettingsPanel::onResetClicked() {
    confidenceSlider_->setValue(50);
    nmsSlider_->setValue(45);
    threadsSpinBox_->setValue(4);
    deviceCombo_->setCurrentIndex(0);
    emit settingsChanged();
}

float SettingsPanel::getConfidenceThreshold() const {
    return confidenceThreshold_;
}

float SettingsPanel::getNMSThreshold() const {
    return nmsThreshold_;
}

int SettingsPanel::getNumThreads() const {
    return threadsSpinBox_->value();
}

QString SettingsPanel::getDevice() const {
    return deviceCombo_->currentData().toString();
}
