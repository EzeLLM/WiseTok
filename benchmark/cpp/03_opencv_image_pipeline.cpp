#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>
#include <cmath>

class ImagePipeline {
private:
    cv::Mat original;
    cv::Mat processed;
    std::vector<cv::Mat> channels;

public:
    ImagePipeline() = default;

    bool loadImage(const std::string& filepath) {
        original = cv::imread(filepath, cv::IMREAD_COLOR);
        if (original.empty()) {
            std::cerr << "Failed to load image: " << filepath << std::endl;
            return false;
        }
        processed = original.clone();
        return true;
    }

    void applyGaussianBlur(int kernel_size = 5, double sigma = 1.0) {
        cv::Mat blurred;
        cv::GaussianBlur(processed, blurred, cv::Size(kernel_size, kernel_size), sigma);
        processed = blurred;
    }

    void applySobelEdgeDetection() {
        cv::Mat gray;
        cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

        cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);

        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, processed);
    }

    void applyCanny(double threshold1 = 50, double threshold2 = 150) {
        cv::Mat gray;
        cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

        cv::Mat edges;
        cv::Canny(gray, edges, threshold1, threshold2);
        processed = edges;
    }

    void findAndDrawContours(int min_area = 100) {
        cv::Mat gray;
        if (processed.channels() == 3) {
            cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = processed.clone();
        }

        cv::Mat binary;
        cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary.clone(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat result = original.clone();
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > min_area) {
                cv::Moments moments = cv::moments(contours[i]);
                cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);
                cv::drawContours(result, contours, (int)i, cv::Scalar(0, 255, 0), 2);
                cv::circle(result, center, 5, cv::Scalar(255, 0, 0), -1);
            }
        }
        processed = result;
    }

    void splitChannels() {
        cv::split(processed, channels);
    }

    void mergeChannels() {
        if (channels.size() == 3) {
            cv::merge(channels, processed);
        }
    }

    void applyHistogramEqualization() {
        cv::Mat gray;
        cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
        cv::Mat equalized;
        cv::equalizeHist(gray, equalized);
        cv::cvtColor(equalized, processed, cv::COLOR_GRAY2BGR);
    }

    void applyMorphology(int operation = cv::MORPH_CLOSE, int kernel_size = 5) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
        cv::Mat morphed;
        cv::morphologyEx(processed, morphed, operation, kernel, cv::Point(-1, -1), 1);
        processed = morphed;
    }

    void applyHoughCircles() {
        cv::Mat gray;
        cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2, 2);

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1, 50,
                        200, 100, 30, 300);

        cv::Mat result = original.clone();
        for (size_t i = 0; i < circles.size(); ++i) {
            cv::Vec3f c = circles[i];
            cv::Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);
            cv::circle(result, center, radius, cv::Scalar(0, 255, 255), 3);
            cv::circle(result, center, 3, cv::Scalar(0, 0, 255), -1);
        }
        processed = result;
    }

    void applyAdaptiveThreshold() {
        cv::Mat gray;
        cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
        cv::Mat binary;
        cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv::THRESH_BINARY, 11, 2);
        processed = binary;
    }

    void drawAnnotations(const std::string& text, const cv::Scalar& color = cv::Scalar(255, 255, 255)) {
        cv::putText(processed, text, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX,
                   1.0, color, 2, cv::LINE_AA);
    }

    void saveImage(const std::string& filepath) {
        cv::imwrite(filepath, processed);
    }

    void displayImage(const std::string& window_name = "Image") {
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name, processed);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat getProcessed() const { return processed; }
    cv::Mat getOriginal() const { return original; }
    int getWidth() const { return processed.cols; }
    int getHeight() const { return processed.rows; }
};

int main(int argc, char** argv) {
    ImagePipeline pipeline;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    if (!pipeline.loadImage(argv[1])) {
        return -1;
    }

    std::cout << "Loaded image: " << pipeline.getWidth() << "x" << pipeline.getHeight() << std::endl;

    pipeline.applyGaussianBlur(5, 1.5);
    pipeline.applyCanny(50, 150);
    pipeline.findAndDrawContours(100);
    pipeline.drawAnnotations("Contours Detected");

    pipeline.saveImage("output_contours.png");

    return 0;
}
