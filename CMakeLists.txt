cmake_minimum_required(VERSION 3.16)
project(yolo_detector_exe VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(OpenCV CONFIG REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime" CACHE PATH "Path to ONNX Runtime directory")

if(WIN32)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
    )
else()
    find_package(onnxruntime REQUIRED)
endif()

set(SOURCES
    src/main.cpp
    src/core/ModelYOLO.cpp
    src/ui/MainWindow.cpp
    src/ui/SettingsPanel.cpp
)

set(HEADERS
    src/ui/MainWindow.h
    src/ui/SettingsPanel.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        onnxruntime
    )
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE
        onnxruntime::onnxruntime
    )
endif()

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
