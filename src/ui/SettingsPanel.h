#ifndef SETTINGSPANEL_H
#define SETTINGSPANEL_H

#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QGroupBox>

class SettingsPanel : public QWidget {
    Q_OBJECT

public:
    explicit SettingsPanel(QWidget *parent = nullptr);
    
    float getConfidenceThreshold() const;
    float getNMSThreshold() const;
    int getNumThreads() const;
    QString getDevice() const;

signals:
    void settingsChanged();

private slots:
    void onConfidenceChanged(int value);
    void onNMSChanged(int value);
    void onApplyClicked();
    void onResetClicked();

private:
    void setupUI();
    
    QSlider* confidenceSlider_;
    QLabel* confidenceLabel_;
    QSlider* nmsSlider_;
    QLabel* nmsLabel_;
    QSpinBox* threadsSpinBox_;
    QComboBox* deviceCombo_;
    QPushButton* applyBtn_;
    QPushButton* resetBtn_;
    
    float confidenceThreshold_;
    float nmsThreshold_;
};

#endif
