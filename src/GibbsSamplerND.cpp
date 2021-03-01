#include <GibbsSamplerND.h>
#include <Logger.h>

#include <Eigen/LU>

#include <algorithm>

GibbsSamplerND::GibbsSamplerND(const Info& info)
: m_u(info.nd.u)
, m_s(info.nd.s)
, m_cond_dists(ComputeConditionalDists())
, m_iter(info.iter)
// , m_standart_dist_sampler({.u = Eigen::VectorXd::Zero(3), .s = Eigen::MatrixXd::Identity(3)})
// , m_gen(m_rd()) {} 
{}

/// WORK WITH PIXELS IS HARDCODED, NEEDS REWORK

// s11 s12
// s21 s22
std::vector<Eigen::MatrixXd> GibbsSamplerND::ComputeConditionalDists() const {
    std::vector<Eigen::MatrixXd> cond_dists(m_u.rows());
    for (int64_t i = 0; i < m_u.rows() / 3; ++i) {
        const auto s22 = RemoveRowCol(m_s, i);
        Eigen::MatrixXd s12 = m_s(Eigen::seq(i * 3, (i + 1) * 3 - 1), Eigen::all);
        RemoveColumn(s12, i);
        cond_dists[i] = s12 * s22.inverse();
    }
    return cond_dists;
}

Eigen::MatrixXd GibbsSamplerND::RemoveRowCol(Eigen::MatrixXd matrix, int64_t i) const {
    RemoveRow(matrix, i);
    RemoveColumn(matrix, i);
    return matrix;
}

void GibbsSamplerND::RemoveRow(Eigen::MatrixXd& matrix, int64_t i) const {
    int64_t new_rows = matrix.rows() - 1 * 3;
    int64_t new_cols = matrix.cols();

    if (i * 3 < new_rows)
        matrix.block(i * 3, 0, new_rows - i * 3, new_cols) = matrix.block((i + 1) * 3, 0, (new_rows - i * 3), new_cols);
    matrix.conservativeResize(new_rows, new_cols);
}

void GibbsSamplerND::RemoveColumn(Eigen::MatrixXd& matrix, int64_t i) const {
    int64_t new_rows = matrix.rows();
    int64_t new_cols = matrix.cols() - 1 * 3;

    if (i * 3 < new_cols)
        matrix.block(0, i * 3, new_rows, new_cols - i * 3) = matrix.block(0, (i + 1) * 3, new_rows, new_cols - i * 3);
    matrix.conservativeResize(new_rows, new_cols);
}

Eigen::VectorXd GibbsSamplerND::SampleWithVals(const Eigen::VectorXd& vals, size_t pos) {
    // std::cout << m_u.rows() << std::endl;
    // std::cout << m_u[pos] << std::endl;
    m_vector = vals;
    const auto u = GetU(pos);
    // std::cout << u << std::endl;
    const auto s = GetS(pos);
    // std::cout << s << std::endl;
    NDSampler sampler({.u = u, .s = s});
    return sampler.Sample();
}


Eigen::VectorXd GibbsSamplerND::Sample() {
    m_vector = m_u;
    for (size_t iter = 0; iter < m_iter; ++iter) {
        UpdateVector();
    }
    return m_vector;
}

void GibbsSamplerND::UpdateVector() {
    for (int64_t i = 0; i < m_u.rows(); ++i) {
        const auto u = GetU(i);
        const auto s = GetS(i);
        // m_vector[i] = SampleND(u, s);
    }
}

Eigen::VectorXd GibbsSamplerND::GetU(int64_t i) const {
    using namespace Eigen;
    const auto vector = GetVectorWithoutI(m_vector, i);
    const auto u_small = GetVectorWithoutI(m_u, i);
    return m_u(seq((i * 3), (i + 1) * 3 - 1)) + m_cond_dists[i] * (vector - u_small);
}

Eigen::MatrixXd GibbsSamplerND::GetS(int64_t i) const {
    using namespace Eigen;
    Eigen::MatrixXd s21 = m_s(Eigen::all, Eigen::seq(i * 3, (i + 1) * 3 - 1));
    RemoveRow(s21, i);
    return m_s(seq(i * 3, (i + 1) * 3 - 1), seq(i * 3, (i + 1) * 3 - 1)) - m_cond_dists[i] * s21;
}

Eigen::VectorXd GibbsSamplerND::GetVectorWithoutI(Eigen::VectorXd vector, int64_t i) const {
    std::copy(vector.begin() + (i + 1) * 3, vector.end(), vector.begin() + i * 3);
    vector.conservativeResize(vector.rows() - 1 * 3);
    return vector;
}

double GibbsSamplerND::SampleND(double u, double s) const {
    return {};
}
