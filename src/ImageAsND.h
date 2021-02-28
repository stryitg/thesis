#pragma once

#include <NormalDistribution.h>
#include <Rect.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ImageAsND {
public:
    
    ImageAsND(const cv::Mat& img, const Tiling& tiling);
    
    NormalDist ComputeND();
    
private:
    std::vector<Eigen::VectorXd> GetNDVectors() const;
    Eigen::VectorXd GetNDVector(size_t i, size_t j) const;
    Eigen::VectorXd EstimateU(const std::vector<Eigen::VectorXd>& vs) const;
    Eigen::MatrixXd EstimateS(const std::vector<Eigen::VectorXd>& vs, const Eigen::VectorXd& u) const;
    
private:
    cv::Mat m_img;
    
    const Tiling m_tiling;
};