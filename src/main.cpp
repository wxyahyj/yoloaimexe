#include <QApplication>
#include "ui/MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    QApplication::setApplicationName("YOLO Detector");
    QApplication::setApplicationVersion("1.0.0");
    QApplication::setOrganizationName("YOLO Project");
    
    MainWindow window;
    window.show();
    
    return app.exec();
}
