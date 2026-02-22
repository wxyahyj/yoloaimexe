#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <memory>
#include "core/ModelYOLO.h"
#include "SettingsPanel.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onStartStopClicked();
    void onSettingsClicked();
    void updateFrame();
    void onModelLoaded();
    void onSettingsChanged();

private:
    void setupUI();
    void loadModel();
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
    cv::Mat convertToRGB(const cv::Mat& frame);

    std::unique_ptr<Ui::MainWindow> ui;
    std::unique_ptr<ModelYOLO> model_;
    std::unique_ptr<cv::VideoCapture> cap_;
    QTimer* timer_;
    QLabel* videoLabel_;
    QPushButton* startStopBtn_;
    QPushButton* settingsBtn_;
    SettingsPanel* settingsPanel_;
    
    bool isRunning_;
    bool modelLoaded_;
};

#endif
