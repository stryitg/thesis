#pragma once

#include <NormalDistribution.h>
#include <Rect.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class NDImageDrawer {
public:
    struct Info {
        
        
        NormalDist nd;
        Rect shape;
        Tiling tiling;
    };
    
    NDImageDrawer(const Info& opt);
    
    cv::Mat Draw(size_t iter) const;

private:
    cv::Mat GenerateRandomFromDist() const;
    void UpdateWithGibbsSampling(cv::Mat& img) const;
    Eigen::VectorXd GetNeighbors(const cv::Mat& img, size_t i, size_t j) const;
    Eigen::VectorXd SamplePixels() const;
    uint8_t ToImageColour(double v) const;

private:
    Info m_info;
};