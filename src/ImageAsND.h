#pragma once

#include <NormalDistribution.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ImageAsND {
public:
    struct Options {
        size_t w;
        size_t h;
    };
    
    ImageAsND(const cv::Mat& img, const Options& opt);
    
    NormalDist ComputeND();
    
private:
    std::vector<Eigen::VectorXd> GetNDVectors() const;
    Eigen::VectorXd GetNDVector(int i, int j) const;
    Eigen::VectorXd EstimateU(const std::vector<Eigen::VectorXd>& vs) const;
    Eigen::MatrixXd EstimateS(const std::vector<Eigen::VectorXd>& vs, const Eigen::VectorXd& u) const;
    
private:
    cv::Mat m_img;
    
    const int m_w;
    const int m_h;
};