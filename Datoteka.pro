TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp
TARGET = camera_feed

# Include directories
INCLUDEPATH += /usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/include/opencv4 /usr/include/eigen3 /usr/lib/x86_64-linux-gnu/glib-2.0/include

# Libraries
LIBS += -lgstreamer-1.0 -lglib-2.0 -lgobject-2.0 -lopencv_core -lopencv_highgui -lopencv_imgproc -lgstapp-1.0
