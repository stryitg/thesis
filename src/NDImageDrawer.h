#pragma once

#include <NormalDistribution.h>
#include <Rect.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class NDImageDrawer {
public:
    struct Info {
        std::vector<NormalDist> nds;
        std::vector<std::vector<std::vector<double>>> blending_coefs;
        Rect shape;
        Tiling tiling;
    };
    
    NDImageDrawer(const Info& opt);
    
    cv::Mat Draw(size_t iter);

private:
    void GenerateRandomFromDist();
    void UpdateWithGibbsSampling();
    Eigen::VectorXd GetNeighbors(size_t i, size_t j) const;
    NormalDist GetBlendedDist(size_t i, size_t j) const;
    Eigen::VectorXd SamplePixels() const;
    uint8_t ToImageColour(double v) const;

    static std::array<double, 3> GetStds(const NormalDist& nd);
    static void NormalizeS(Eigen::MatrixXd& s, const std::array<double, 3>& stds);
    static std::array<double, 3> GetUs(const Eigen::VectorXd& us);

    void ShowImg() const;

private:    
    Info m_info;
    std::vector<std::vector<std::array<double, 3>>> m_img;

    std::array<double, 3> m_stds;
    std::array<double, 3> m_us;
};