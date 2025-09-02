#include <iostream>
#include <gst/gst.h>
#include <opencv2/opencv.hpp>
#include <gst/app/gstappsink.h>
#include <thread>
#include <atomic>
#include <ctime>
#include <mutex>
#include <queue>
#include <condition_variable>

std::atomic<bool> running(true);
std::atomic<bool> closeAllWindows(false); // Flag to indicate if all windows should be closed
std::atomic<bool> roiImageDisplayed(false); // Flag to indicate if ROI Image has been displayed

struct ROI {
    int left_closest, left_furthest, top_closest, top_furthest;
};

class ROIImage {
private:
    cv::Mat image;
public:
    void setImage(const cv::Mat& img) {
        image = img.clone();
    }
    cv::Mat getImage() const {
        return image;
    }
};

class PixelMMData {
public:
    int furthestPixelDepth;
    int lineDistance;
    double realWorldDistanceMM;

    PixelMMData() : furthestPixelDepth(-1), lineDistance(-1), realWorldDistanceMM(-1.0) {}

    void updateData(int furthestDepth, int lineDist, double realDistMM) {
        furthestPixelDepth = furthestDepth;
        lineDistance = lineDist;
        realWorldDistanceMM = realDistMM;
    }
};

ROIImage roiImage;
PixelMMData pixelMMData;

std::queue<int> keyPressQueue;
std::mutex keyPressMutex;
std::condition_variable keyPressCV;

void keyPressHandler() {
    while (running) {
        int key = cv::waitKey(10); // Wait for 10 milliseconds
        if (key != -1) { // Key press detected
            std::lock_guard<std::mutex> lock(keyPressMutex);
            keyPressQueue.push(key);
            keyPressCV.notify_one();
        }
    }
}

GstFlowReturn onNewSample(GstElement *sink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) {
        std::cerr << "Error: Failed to pull sample from appsink." << std::endl;
        return GST_FLOW_ERROR;
    }
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        std::cerr << "Error: Failed to pull buffer from appsink." << std::endl;
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    GstCaps *caps = gst_sample_get_caps(sample);
    if (!caps) {
        std::cerr << "Error: Failed to pull caps from appsink." << std::endl;
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    GstStructure *structure = gst_caps_get_structure(caps, 0);
    if (!structure) {
        std::cerr << "Error: Failed to pull structure from appsink." << std::endl;
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    // Extract width and height of video frame
    int width, height;
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);

    // Conversion GStreamer to OpenCV Mat
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);
    cv::Mat frame(height, width, CV_8UC3, (uchar*)map.data); // Use CV_8UC3 for BGR format
    gst_buffer_unmap(buffer, &map);

    // Update "Camera Feed" window in the background
    if (!roiImageDisplayed) {
        cv::imshow("Camera Feed", frame);
    }

    // Pixel reader for ROI
    ROI roi;
    roi.left_closest = width; // Initialize to maximum possible
    roi.left_furthest = 0; // Initialize to minimum possible
    roi.top_closest = height; // Initialize to maximum possible
    roi.top_furthest = 0; // Initialize to minimum possible

    int intensityThreshold = 40;

    for (int y = 110; y < std::min(900, height); ++y) { // Limit y to 900 pixels
        for (int x = 500; x < 1520; ++x) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);
            if (pixel[0] > intensityThreshold || pixel[1] > intensityThreshold || pixel[2] > intensityThreshold) { // Non-black pixel
                // Update ROI
                if (x < roi.left_closest) {
                    roi.left_closest = x;
                }
                if (x > roi.left_furthest) {
                    roi.left_furthest = x;
                }
                if (y < roi.top_closest) {
                    roi.top_closest = y;
                }
                if (y > roi.top_furthest) {
                    roi.top_furthest = y;
                }
            }
        }
    }

    // Create ROI image
    int roiHeight = std::min(900, height) - roi.top_closest; // Limit ROI height

    //Define the left and right boundaries of ROI
    int roi_left_boundary = roi.left_closest;
    int roi_right_boundary = roi.left_furthest;

    //Adjust the left boundary to the closest non-black pixel
    while (roi_left_boundary > 500) {
        cv::Vec3b pixel = frame.at<cv::Vec3b>(roi.top_closest, roi_left_boundary - 1);
        if (pixel[0] <= intensityThreshold && pixel[1] <= intensityThreshold && pixel[2] <= intensityThreshold) {
            break; //found the boundary
        }
        roi_left_boundary--;
    }

    //Adjust the right boundary to the furthest non-black pixel
    while (roi_right_boundary < 1520) {
        cv::Vec3b pixel = frame.at<cv::Vec3b>(roi.top_closest, roi_right_boundary + 1);
        if (pixel[0] <= intensityThreshold && pixel[1] <= intensityThreshold && pixel[2] <= intensityThreshold) {
            break; //found the boundary
        }
        roi_right_boundary++;
    }

    //Create ROI Image
    int roiWidth = roi_right_boundary - roi_left_boundary;
    cv::Rect roiRect(roi_left_boundary, roi.top_closest, roiWidth, roiHeight);
    cv::Mat roiImg = frame(roiRect).clone();
    roiImage.setImage(roiImg);

    // Calculate the depth of the furthest non-white pixel from the top within the ROI
    int furthest_pixel_depth = roi.top_furthest;
    for (int y = roi.top_closest; y <= roi.top_furthest; ++y) {
        for (int x = roi_left_boundary; x < roi_right_boundary - 28; ++x) { // Ignore rightmost 28 pixels
            cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);
            if (pixel[0] > intensityThreshold || pixel[1] > intensityThreshold || pixel[2] > intensityThreshold) { // Non-white pixel
                furthest_pixel_depth = y;
                break;
            }
        }
    }

    furthest_pixel_depth -= 110; // Adjust depth calculation

    // Reader for white horizontal lines and red dots with thresholds
    std::vector<int> line_positions; // Store positions of white lines and red dots
    int white_threshold = 200; // Threshold for white lines
    int red_threshold = 10; // Threshold for red dots

    // Find positions of white lines and red dots in the original search area
    for (int y = 0; y < roiImage.getImage().rows; ++y) {
        cv::Vec3b pixel = roiImage.getImage().at<cv::Vec3b>(y, roiImage.getImage().cols - 15); // Start from the 10th to 20th pixels from the right
        if ((pixel[0] >= white_threshold && pixel[1] >= white_threshold && pixel[2] >= white_threshold) ||
            (pixel[2] >= red_threshold && pixel[0] < red_threshold && pixel[1] < red_threshold)) { // White pixel within threshold range or red dot
            line_positions.push_back(y);
        }
    }

    // If no white lines or red dots found, search in the alternate area
    if (line_positions.empty()) {
        for (int y = 0; y < roiImage.getImage().rows; ++y) {
            cv::Vec3b pixel = roiImage.getImage().at<cv::Vec3b>(y, roiImage.getImage().cols - 10); // Start from the 4th to 10th pixels from the right
            if ((pixel[0] >= white_threshold && pixel[1] >= white_threshold && pixel[2] >= white_threshold) ||
                (pixel[2] >= red_threshold && pixel[0] < red_threshold && pixel[1] < red_threshold)) { // White pixel within threshold range or red dot
                line_positions.push_back(y);
            }
        }
    }

    // Group closely spaced lines
    std::vector<int> line_groups; // Store positions of line groups
    int group_threshold = 5; // Adjust this threshold as needed
    for (size_t i = 0; i < line_positions.size(); ++i) {
        bool grouped = false;
        for (int& group : line_groups) {
            if (std::abs(line_positions[i] - group) <= group_threshold) {
                group = (line_positions[i] + group) / 2; // Update group position to the average
                grouped = true;
                break;
            }
        }
        if (!grouped) {
            line_groups.push_back(line_positions[i]);
        }
    }

    // Calculate pixel distance between the line groups
    int line_distance = -1;
    if (line_groups.size() >= 2) {
        line_distance = line_groups[1] - line_groups[0];
    }

    // Define the pixel-to-millimeter ratio
    double pixel_to_mm_ratio = 0.2645833333; // 1 pixel distance corresponds to 0.2645833333 mm

    // Calculate real-world distance based on line distance and pixel-to-millimeter ratio
    double real_world_distance_mm = -1; // Default value
    if (line_distance != -1 && pixel_to_mm_ratio != -1) {
        real_world_distance_mm = line_distance * pixel_to_mm_ratio;
    }

    // Update pixelMMData object
    pixelMMData.updateData(furthest_pixel_depth, line_distance, real_world_distance_mm);

    // Update window title with furthest non-white pixel depth and ROI dimensions
    std::string windowTitleDepthDimensions = "Depth: " + std::to_string(pixelMMData.furthestPixelDepth) + " | Dimensions: " + std::to_string(roiWidth) + "x" + std::to_string(roiHeight);
    cv::putText(roiImage.getImage(), windowTitleDepthDimensions, cv::Point(10, roiImage.getImage().rows - 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Update window title with line distance and real-world distance
    std::string windowTitleLineDistanceRealWorld = "Pxl distance: ";
    if (pixelMMData.lineDistance != -1) {
        windowTitleLineDistanceRealWorld += std::to_string(pixelMMData.lineDistance) + " pixels";
    } else {
        windowTitleLineDistanceRealWorld += "N/A";
    }
    windowTitleLineDistanceRealWorld += " | Distance: ";
    if (pixelMMData.realWorldDistanceMM != -1) {
        windowTitleLineDistanceRealWorld += std::to_string(pixelMMData.realWorldDistanceMM) + " mm";
    } else {
        windowTitleLineDistanceRealWorld += "N/A";
    }
    cv::putText(roiImage.getImage(), windowTitleLineDistanceRealWorld, cv::Point(10, roiImage.getImage().rows - 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Display ROI
    // Display ROI image only once
    if (!roiImageDisplayed) {
        cv::imshow("ROI Image", roiImage.getImage());
        roiImageDisplayed = true; // Set flag to indicate ROI Image has been displayed
    } else {
        // Update ROI Image
        cv::imshow("ROI Image", roiImage.getImage());
    }

    // Cleaning
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Create GStreamer pipeline to capture video from the camera
    GstElement *pipeline = gst_parse_launch("v4l2src device=/dev/video0 io-mode=2 ! video/x-raw,format=YUY2,width=1920,height=1080 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink max-buffers=1 drop=true sync=false emit-signals=true", NULL);
    if (!pipeline) {
        std::cerr << "Error: Failed to create GStreamer pipeline." << std::endl;
        return -1;
    }

    // Get the appsink element from the pipeline
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsink) {
        std::cerr << "Error: Failed to get appsink element from the pipeline." << std::endl;
        gst_object_unref(pipeline);
        return -1;
    }

    // Setting appsink to get OpenCV compatible buffers
    g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(onNewSample), NULL);

    // Start the pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Create OpenCV window to display the feed (hidden)
    cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
    cv::moveWindow("Camera Feed", -1000, -1000); // Move window outside the screen bounds

    // Create OpenCV window to display the ROI image
    cv::namedWindow("ROI Image", cv::WINDOW_NORMAL);

    // Start the thread for key press detection
    std::thread keyThread(keyPressHandler);

    // Main loop for processing frames
    while (running) {
        g_main_context_iteration(NULL, TRUE);

        // Check for key presses
        {
            std::unique_lock<std::mutex> lock(keyPressMutex);
            if (!keyPressQueue.empty()) {
                int key = keyPressQueue.front();
                keyPressQueue.pop();
                lock.unlock();

                // Process key press
                if (key == 'q') {
                    std::cout << "Detected 'q' key press." << std::endl;
                    running = false;
                    closeAllWindows = true; // Set flag to close all windows
                } else if (key == 'p') {
                    std::cout << "Detected 'p' key press. Saving ROI Image..." << std::endl;
                    // Generate file name with current timestamp
                    std::time_t currentTime = std::time(nullptr);
                    std::string filename = "/home/user/Vision_app/camera_feed/Pictures/roi_image_" + std::to_string(currentTime) + ".jpg";
                    // Save ROI Image
                    cv::imwrite(filename, roiImage.getImage());
                    std::cout << "ROI Image saved as: " << filename << std::endl;
                }
            }
        }

        // Check if all windows should be closed
        if (closeAllWindows) {
            cv::destroyAllWindows(); // Close all OpenCV windows
            break; // Exit the loop if all windows should be closed
        }
    }

    // Clean up
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);

    // Join key press handler thread
    if (keyThread.joinable()) {
        keyThread.join();
    }

    return 0;
}
