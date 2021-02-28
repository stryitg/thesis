#include <ImageAsND.h>
#include <Logger.h>
#include <Eigen/Eigenvalues>

ImageAsND::ImageAsND(const cv::Mat& img, const Options& opt)
: m_img(img)
, m_w(static_cast<int>(opt.w))
, m_h(static_cast<int>(opt.h)) {}

NormalDist ImageAsND::ComputeND() {
    const auto vals = GetNDVectors();
    const auto u = EstimateU(vals);
    const auto s = EstimateS(vals, u);
    return {.u = u, .s = s};
}

std::vector<Eigen::VectorXd> ImageAsND::GetNDVectors() const {
    std::vector<Eigen::VectorXd> vals((m_img.rows - 2 * m_h) * (m_img.cols - 2 * m_w),
                                      Eigen::VectorXd((2 * m_h + 1) * (2 * m_w + 1) * 3));
    size_t count = 0;
    for (int i = m_h; i < m_img.rows - m_h; ++i) {
        for (int j = m_w; j < m_img.cols - m_w; ++j) {
            vals[count++] = GetNDVector(i, j);
        }
    }
    return vals;
}

Eigen::VectorXd ImageAsND::GetNDVector(int i, int j) const {
    size_t count = 0;
    auto nd_vec = Eigen::VectorXd((2 * m_h + 1) * (2 * m_w + 1) * 3);
    for (int k = i - m_h; k < i + m_h + 1; ++k) {
        for (int m = j - m_w; m < j + m_w + 1; ++m) {
            const auto val = m_img.at<cv::Vec3b>(k, m);
            nd_vec[count++] = val[0];
            nd_vec[count++] = val[1];
            nd_vec[count++] = val[2];
        }
    }
    return nd_vec;
}

Eigen::VectorXd ImageAsND::EstimateU(const std::vector<Eigen::VectorXd>& vs) const {
    LOG("Estimating avg");
    Eigen::VectorXd m = Eigen::VectorXd::Zero(vs[0].rows());
    for (const auto v : vs) {
        m += v;
    }
    LOG("Done Estimating avg");
    return m / vs.size();
}

Eigen::MatrixXd ImageAsND::EstimateS(const std::vector<Eigen::VectorXd>& vs, const Eigen::VectorXd& u) const {
    LOG("Estimating sigma");
    Eigen::MatrixXd m = Eigen::ArrayXXd::Zero(vs[0].rows(), vs[0].rows());
    for (const auto v : vs) {
        m += (v - u) * (v - u).transpose();
    }
    LOG("Done Estimating sigma");
    const auto es = m.eigenvalues();
    LOG(es);
    return m / (vs.size() - 1);
}
