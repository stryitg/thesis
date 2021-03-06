#pragma once

#include <NormalDistribution.h>
#include <NDSampler.h>

#include <cstdint>
#include <random>

class GibbsSamplerND {
public:
    struct Info {
        NormalDist nd;
        size_t iter;
    };
    
    GibbsSamplerND(const Info& info);
    
    Eigen::VectorXd Sample();
    Eigen::VectorXd SampleWithVals(const Eigen::VectorXd& vals, size_t pos);
    
private:
    std::vector<Eigen::MatrixXd> ComputeConditionalDists() const;
    Eigen::MatrixXd RemoveRowCol(Eigen::MatrixXd matrix, int64_t i) const;
    void RemoveRow(Eigen::MatrixXd& matrix, int64_t i) const;
    void RemoveColumn(Eigen::MatrixXd& matrix, int64_t i) const;
    
    void UpdateVector();
    Eigen::VectorXd GetU(int64_t i) const;
    Eigen::MatrixXd GetS(int64_t i) const;
    Eigen::VectorXd GetVectorWithoutI(Eigen::VectorXd vector, int64_t i) const;
    double SampleND(double u, double s) const;
    
private:
    Eigen::VectorXd m_u;
    Eigen::MatrixXd m_s;
    
    std::vector<Eigen::MatrixXd> m_cond_dists;
    
    const size_t m_iter;
    
    Eigen::VectorXd m_vector;
    
    // mutable std::random_device m_rd;
    // mutable std::mt19937 m_gen;
    // mutable NDSampler m_standart_dist_sampler;
};