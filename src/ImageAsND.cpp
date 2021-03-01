#include <ImageAsND.h>
#include <Logger.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <execution>

ImageAsND::ImageAsND(const cv::Mat& img, const Tiling& tiling)
: m_img(img)
, m_tiling(tiling) {}

NormalDist ImageAsND::ComputeND() {
    const auto vals = GetNDVectors();
    const auto u = EstimateU(vals);
    const auto s = EstimateS(vals, u);
    return {.u = u, .s = s};
}

std::vector<Eigen::VectorXd> ImageAsND::GetNDVectors() const {
    std::vector<Eigen::VectorXd> vals((m_img.rows - 2 * m_tiling.r_h) * (m_img.cols - 2 * m_tiling.r_w),
                                      Eigen::VectorXd((2 * m_tiling.r_h + 1) * (2 * m_tiling.r_w + 1) * 3));
    size_t count = 0;
    for (size_t i = m_tiling.r_h; i < m_img.rows - m_tiling.r_h; ++i) {
        for (size_t j = m_tiling.r_w; j < m_img.cols - m_tiling.r_w; ++j) {
            vals[count++] = GetNDVector(i, j);
        }
    }
    return vals;
}

Eigen::VectorXd ImageAsND::GetNDVector(size_t i, size_t j) const {
    size_t count = 0;
    auto nd_vec = Eigen::VectorXd((2 * m_tiling.r_h + 1) * (2 * m_tiling.r_w + 1) * 3);
    for (size_t k = i - m_tiling.r_h; k < i + m_tiling.r_h + 1; ++k) {
        for (size_t m = j - m_tiling.r_w; m < j + m_tiling.r_w + 1; ++m) {
            const auto& val = m_img.at<cv::Vec3b>(static_cast<int>(k), static_cast<int>(m));
            nd_vec[count++] = val[0];
            nd_vec[count++] = val[1];
            nd_vec[count++] = val[2];
            
        }
    }
    return nd_vec;
}

Eigen::VectorXd ImageAsND::EstimateU(const std::vector<Eigen::VectorXd>& vs) const {
    std::cout << "Estimating avg" << std::endl;
    Eigen::VectorXd m = Eigen::VectorXd::Zero(vs[0].rows());
    for (const auto& v : vs) {
        m += v;
    }
    std::cout << "Done Estimating avg" << std::endl;
    return m / vs.size();
}

Eigen::MatrixXd ImageAsND::EstimateS(const std::vector<Eigen::VectorXd>& vs, const Eigen::VectorXd& u) const {
    std::cout << "Estimating sigma" << std::endl;
    Eigen::MatrixXd m = Eigen::ArrayXXd::Zero(vs[0].rows(), vs[0].rows());
    // std::vector<Eigen::MatrixXd> sigmas(vs.size());
    // std::transform(std::execution::par, vs.begin(), vs.end(), sigmas.begin(), [&u] (const auto& v) {
    //     return (v - u) * (v - u).transpose(); 
    // });
    // const auto m = std::reduce(std::execution::par, sigmas.begin(), sigmas.end());
    size_t count = 0;
    for (const auto& v : vs) {
        if (count++ % 10000 == 0)
            std::cout << count - 1 << std::endl;
        m += (v - u) * (v - u).transpose();
    }
    // std::cout << "Done Estimating sigma" << std::endl;
    return m / (vs.size() - 1);
}
