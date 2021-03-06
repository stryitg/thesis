#include <GibbsSamplerND.h>
#include <Logger.h>

#include <Eigen/LU>

#include <algorithm>

GibbsSamplerND::GibbsSamplerND(const Info& info)
: m_u(info.nd.u)
, m_s(info.nd.s)
, m_cond_dists(ComputeConditionalDists())
, m_iter(info.iter)
, m_gen(m_rd()) {} 

// s11 s12
// s21 s22
std::vector<Eigen::VectorXd> GibbsSamplerND::ComputeConditionalDists() const {
    std::vector<Eigen::VectorXd> cond_dists(m_u.rows());
    for (int64_t i = 0; i < m_u.rows(); ++i) {
        const auto s22 = RemoveRowCol(m_s, i);
        const auto s12 = GetVectorWithoutI(m_s.row(i), i);
        cond_dists[i] = s12.transpose() * s22.inverse();
    }
    return cond_dists;
}

Eigen::MatrixXd GibbsSamplerND::RemoveRowCol(Eigen::MatrixXd matrix, int64_t i) const {
    RemoveRow(matrix, i);
    RemoveColumn(matrix, i);
    return matrix;
}

void GibbsSamplerND::RemoveRow(Eigen::MatrixXd& matrix, int64_t i) const {
    int64_t new_rows = matrix.rows() - 1;
    int64_t new_cols = matrix.cols();

    if (i < new_rows)
        matrix.block(i, 0, new_rows - i, new_cols) = matrix.block(i + 1, 0, new_rows - i, new_cols);
    matrix.conservativeResize(new_rows, new_cols);
}

void GibbsSamplerND::RemoveColumn(Eigen::MatrixXd& matrix, int64_t i) const {
    int64_t new_rows = matrix.rows();
    int64_t new_cols = matrix.cols() - 1;

    if (i < new_cols)
        matrix.block(0, i, new_rows, new_cols - i) = matrix.block(0, i + 1, new_rows, new_cols - i);
    matrix.conservativeResize(new_rows,new_cols);
}

double GibbsSamplerND::SampleWithVals(const Eigen::VectorXd& vals, size_t pos) {
    m_vector = vals;
    const auto u = GetU(pos);
    const auto s = GetS(pos);
    // std::cout << u << std::endl;
    return SampleND(u, s);
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
        m_vector[i] = SampleND(u, s);
    }
}

double GibbsSamplerND::GetU(int64_t i) const {
    const auto vector = GetVectorWithoutI(m_vector, i);
    const auto u_small = GetVectorWithoutI(m_u, i);
    return m_u[i] + m_cond_dists[i].dot(vector - u_small);
}

double GibbsSamplerND::GetS(int64_t i) const {
    const auto s21 = GetVectorWithoutI(m_s.col(i), i);
    return m_s(i, i) - m_cond_dists[i].dot(s21);
}

Eigen::VectorXd GibbsSamplerND::GetVectorWithoutI(Eigen::VectorXd vector, int64_t i) const {
    std::copy(vector.begin() + i + 1, vector.end(), vector.begin() + i);
    vector.conservativeResize(vector.rows() - 1);
    return vector;
}

double GibbsSamplerND::SampleND(double u, double s) const {
    std::normal_distribution<double> d(u, std::sqrt(s));
    return d(m_gen);
}
