#pragma once

#include <NormalDistribution.h>
#include <random>


class NDSampler {
public:
    NDSampler(const NormalDist& nd);
    
    Eigen::VectorXd Sample() const;
    
private:
    Eigen::VectorXd GetStandartNDVec() const;
    
private:
    NormalDist m_nd;
    Eigen::MatrixXd m_s_sqrt;
    
    mutable std::random_device m_rd;
    mutable std::mt19937 m_gen;
    mutable std::normal_distribution<double> m_d;
};