#include "net.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

static void draw_objects(const cv::Mat &bgr, const cv::Mat &fgr, const cv::Mat &pha, cv::Mat &comp) {

    // cv::Mat comp;
    cv::resize(bgr, comp, pha.size(), 0, 0, 1);
    for(int i = 0; i < pha.rows; i++) {
        for(int j = 0; j < pha.cols; j++) {
            uchar data = pha.at<uchar>(i, j);
            float alpha = (float)data / 255;
            comp.at<cv::Vec3b>(i, j)[0] = fgr.at<cv::Vec3b>(i, j)[0] * alpha + (1 - alpha) * 255;
            comp.at<cv::Vec3b>(i, j)[1] = fgr.at<cv::Vec3b>(i, j)[1] * alpha + (1 - alpha) * 155;
            comp.at<cv::Vec3b>(i, j)[2] = fgr.at<cv::Vec3b>(i, j)[2] * alpha + (1 - alpha) * 120;
        }
    }

    cv::imshow("pha", pha);   // alphaチャンネル
    cv::imshow("fgr", fgr);   // 合成前の画像
    cv::imshow("comp", comp); // 合成画像
    const int key = cv::waitKey(10);
}

static int detect_rvm(ncnn::Net &net, const cv::Mat &bgr, cv::Mat &pha, cv::Mat &fgr) {
    const int target_width = 512;
    const int target_height = target_width;

    ncnn::Extractor ex = net.create_extractor();
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
    ncnn::Mat ncnn_in1 = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_width, target_height);

    ncnn_in1.substract_mean_normalize(mean_vals, norm_vals);
    ex.input("in0", ncnn_in1);
    ncnn::Mat pha_;
    ncnn::Mat fgr_;
    ex.extract("out0", fgr_);
    ex.extract("out1", pha_);

    cv::Mat cv_pha = cv::Mat(pha_.h, pha_.w, CV_32FC1, (float *)pha_.data);
    cv::Mat cv_fgr = cv::Mat(fgr_.h, fgr_.w, CV_32FC3);
    ncnn::Mat fgr_pack3;
    ncnn::convert_packing(fgr_, fgr_pack3, 3);
    memcpy((uchar *)cv_fgr.data, fgr_pack3.data, 512 * 512 * 3 * sizeof(float));

    resize(cv_pha, cv_pha, cv::Size(bgr.cols, bgr.rows), cv::INTER_LINEAR);
    resize(cv_fgr, cv_fgr, cv::Size(bgr.cols, bgr.rows), cv::INTER_LINEAR);

    cv::Mat fgr8U;
    cv_fgr.convertTo(fgr8U, CV_8UC3, 255.0, 0);

    cv::Mat pha8U;
    cv_pha.convertTo(pha8U, CV_8UC1, 255.0, 0);

    cv::cvtColor(fgr8U, fgr8U, cv::COLOR_BGR2RGB);

    pha8U.copyTo(pha);
    fgr8U.copyTo(fgr);

    return 0;
}

int main(int argc, char **argv) {

    cv::VideoCapture cap("/dev/video2");

    if(!cap.isOpened()) {
        std::cout << "Cannot open cam" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::CAP_OPENCV_MJPEG);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    // original pretrained model from https://github.com/PeterL1n/RobustVideoMatting
    net.load_param("ts.ncnn.param");
    net.load_model("ts.ncnn.bin");

    while(cap.isOpened()) {
        cv::Mat frame, comp;
        cv::Mat fgr, pha;

        cap >> frame;

        cv::Mat fgr, pha, comp;
        detect_rvm(net, frame, fgr, pha);
        draw_objects(frame, fgr, pha, comp); // 后处理
    }

    return 0;
}
