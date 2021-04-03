#pragma once

#include <NormalDistribution.h>
#include <random>
#include <memory>


class NDSampler {
public:
    NDSampler(const NormalDist& nd);
    // NDSampler(const NDSampler&) = default;
    NDSampler(NDSampler&&) = default;
    
    Eigen::VectorXd Sample() const;
    
private:
    Eigen::VectorXd GetStandartNDVec() const;
    
private:
    struct RNG {
        RNG()
        : rd()
        , gen(rd()) {}
        
        std::random_device rd;
        std::mt19937 gen;
        std::normal_distribution<double> d;
    };
    
    NormalDist m_nd;
    Eigen::MatrixXd m_s_sqrt;
    
    mutable std::unique_ptr<RNG> m_rng;
    
};