#include <NDSampler.h>
#include <cassert>
#include <Eigen/Eigenvalues>
#include <iostream>

NDSampler::NDSampler(const NormalDist& nd)
: m_nd(nd)
, m_gen(m_rd()) {
    assert(nd.u.rows() == nd.s.rows());
    assert(nd.s.rows() == nd.s.cols());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m_nd.s);
    m_s_sqrt = es.operatorSqrt();

}

Eigen::VectorXd NDSampler::Sample() const {
    const auto standart_nd_vec = GetStandartNDVec();
    return m_nd.u + m_s_sqrt * standart_nd_vec;
}

Eigen::VectorXd NDSampler::GetStandartNDVec() const {
    auto vec = Eigen::VectorXd(m_nd.u.rows());
    for (auto& v : vec) {
        v = m_d(m_gen);
    }
    return vec;
}