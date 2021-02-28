#pragma once

#include <NormalDistribution.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class NDImageDrawer {
public:
    struct Options {
        NormalDist nd;
        size_t w;
        size_t h;
        size_t r_w;
        size_t r_h;
    };
    
    NDImageDrawer(const Options& opt);
    
    cv::Mat Draw() const;

private:
    Eigen::VectorXd SamplePixels() const;
    uint8_t ToImageColour(double v) const;

private:
    NormalDist m_nd;
    size_t m_w;
    size_t m_h;
    size_t m_r_w;
    size_t m_r_h;
};