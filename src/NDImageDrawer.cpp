#include <NDImageDrawer.h>
#include <GibbsSamplerND.h>
#include <NDSampler.h>
#include <Logger.h>

NDImageDrawer::NDImageDrawer(const Info& info)
: m_info(info)
, m_img(info.shape.h, std::vector<std::array<double, 3>>(info.shape.w,
        std::array<double, 3>({0, 0, 0}))) {}

std::array<double, 3> NDImageDrawer::GetStds(const NormalDist& nd) {
    std::array<double, 3> stds = {0, 0, 0};
    for (long i = 0; i < nd.s.rows() / 3; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            stds[c] += nd.s(i * 3 + c,  i * 3 + c);
        }
    }
    for (size_t c = 0; c < 3; ++c) {
        stds[c] /= (nd.s.rows() / 3);
        stds[c] = std::sqrt(stds[c]);
        std::cout << stds[c] << std::endl;
    }
    return stds;
}

void NDImageDrawer::NormalizeS(Eigen::MatrixXd& s, const std::array<double, 3>& stds) {
    for (long i = 0; i < s.rows(); ++i) {
        for (long j = 0; j < s.cols(); ++j) {
            s(i, j) /= stds[i % stds.size()] * stds[j % stds.size()];
        }
    }
}

std::array<double, 3> NDImageDrawer::GetUs(const Eigen::VectorXd& u) {
    std::array<double, 3> us = {0, 0, 0};
    for (long i = 0; i < u.rows() / 3; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            us[c] += u(i * 3 + c) / (u.rows() / 3);
        }
    }
    return us;
}


void NDImageDrawer::ShowImg() const {
    // std::array<double, 3> us = {60.94715569, 98.27298802, 26.253};
    // std::array<double, 3> stds = {29.27578179, 30.15633032, 23.83153467};
    cv::Mat out(m_img.size(), m_img[0].size(), CV_8UC3);
    for (size_t i = 0; i < m_img.size(); ++i) {
        for (size_t j = 0; j < m_img[0].size(); ++j) {
            const auto& in_val = m_img[i][j];
            auto& out_val = out.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
            for (size_t c = 0; c < 3; ++c) {
                out_val[c] = ToImageColour(in_val[c]);
                // if (ToImageColour(in_val[c]) != static_cast<uint8_t>(in_val[c])) {
                //     std::cout << in_val << std::endl;
                // } 
            }
        }
    }
    cv::imshow("result", out);
    cv::waitKey();
}

cv::Mat NDImageDrawer::Draw(size_t iter) {
    GenerateRandomFromDist();
    // ShowImg();
    for (size_t i = 0; i < iter; ++i) {
        std::cout << i << std::endl;
        UpdateWithGibbsSampling();
        if (i == 10 )
            ShowImg();
    }
    // ShowImg(img);
    // ShowImg(img);
    // return img;
    return {};
}

void NDImageDrawer::GenerateRandomFromDist() {
    const auto tiling_h = m_info.tiling.r_h * 2 + 1;
    const auto tiling_w = m_info.tiling.r_w * 2 + 1;
    std::vector<NDSampler> samplers;
    for (size_t s = 0; s < m_info.nds.size(); ++s) {
        samplers.emplace_back(m_info.nds[s]);
    }
    for (size_t i = 0; i < m_info.shape.h / tiling_h; ++i) {
        for (size_t j = 0; j < m_info.shape.w / tiling_w; ++j) {
            for (size_t s = 0; s < samplers.size(); ++s) {
                const auto vals = samplers[s].Sample();
                size_t count = 0;
                for (size_t k = 0; k < tiling_h; ++k) {
                    for (size_t m = 0; m < tiling_w; ++m) {
                        const auto pos_h = i * tiling_h + k;
                        const auto pos_w = j * tiling_w + m;
                        auto& img_val = m_img[pos_h][pos_w];
                        for (size_t c = 0; c < 3; ++c) {
                            // if (pos_h == m_info.shape.h - 1 || pos_w == m_info.shape.w - 1
                            //     || pos_h == 0 || pos_w == 0)
                                // img_val[c] = m_info.nds[s].u[count++];
                            // else
                                img_val[c] += m_info.blending_coefs[pos_h][pos_w][s] * vals[count++];
                        }
                        
                            
                        // img_val = {255, 255, 255};
                    }
                }
            }
    
        }
    }
}

void NDImageDrawer::UpdateWithGibbsSampling() {
    // GibbsSamplerND sampler({.nd = m_info.nd, .iter = 0});
    const auto pos = (2 * m_info.tiling.r_h + 1) * (2 * m_info.tiling.r_w + 1) / 2;
    // const auto vals = GetNeighbors(img, 2, 2);
    // std::cout << vals << std::endl;
    // sampler.SampleWithVals(vals, pos);
    
    // std::cout << "[";
    // for (const auto& v : vals) {
    //     std::cout << v << ", ";
    // }
    // std::cout << "]" << std::endl;
    
    // for (size_t c = 0; c < 1; ++c) {
    // const auto new_pixel = sampler.SampleWithVals(vals, pos);
    // std::cout << new_pixel << std::endl;
        // std::cout << vals[0] << " "<< vals[1] <<  " "<< vals[2] << std::endl;
        // auto& img_val = img.at<cv::Vec3b>(static_cast<int>(0), static_cast<int>(0));
        // // std::cout << "before: " << (int) img_val[0] << " " << (int) img_val[1] << " "  << (int) img_val[2] << std::endl;
        // img_val[c] = ToImageColour(sampler.SampleWithVals(vals, 0)); 
        // std::cout << "after: " << (int) img_val[0] << " "  << (int) img_val[1] << " "  << (int) img_val[2] << std::endl;
    // }
    // auto& img_val = img.at<cv::Vec3b>(static_cast<int>(0), static_cast<int>(0));
    // std::cout << img_val << std::endl;
    // std::cout << pos << std::endl;
    for (size_t i = 0; i < m_info.shape.h; ++i) {
        for (size_t j = 0; j < m_info.shape.w; ++j) {
            GibbsSamplerND sampler({.nd = GetBlendedDist(i, j), .iter = 0});
            // auto& img_val = img.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
            // std::cout << "BEFORE: " <<  img_val << std::endl;
            const auto vals = GetNeighbors(i, j);
            const auto new_pixel = sampler.SampleWithVals(vals, pos);
            auto& img_val = m_img[i][j];
            for (size_t c = 0; c < 3; ++c) {
                assert(vals(pos * 3 + c) == img_val[c]);
                // std::cout << "before: " << (int) img_val[0] << " " << (int) img_val[1] << " "  << (int) img_val[2] << std::endl;
                img_val[c] = new_pixel[c];
                // std::cout << "after: " << (int) img_val[0] << " "  << (int) img_val[1] << " "  << (int) img_val[2] << std::endl;
            }
            // std::cout << img_val << std::endl;
            // std::cout << "AFTER: " <<  img_val << std::endl;
        }
    }
}

Eigen::VectorXd NDImageDrawer::GetNeighbors(size_t i, size_t j) const {
    size_t count = 0;
    auto nd_vec = Eigen::VectorXd((2 * m_info.tiling.r_h + 1) * (2 * m_info.tiling.r_w + 1) * 3);
    for (int k = i - m_info.tiling.r_h; k < (int) (i + m_info.tiling.r_h + 1); ++k) {
        for (int m = j - m_info.tiling.r_w; m < (int) (j + m_info.tiling.r_w + 1); ++m) {
            const size_t k_ = k >= 0 ? k % m_info.shape.h : m_info.shape.h + k;
            const size_t m_ = m >= 0 ? m % m_info.shape.w : m_info.shape.w + m; 
            const auto& val = m_img[k_][m_];
            for (size_t c = 0; c < 3; ++c) {
                nd_vec[count++] = val[c];
            }
        }
    }
    assert(count == static_cast<size_t>(nd_vec.rows()));
    return nd_vec;
}

NormalDist NDImageDrawer::GetBlendedDist(size_t i, size_t j) const {
    const auto sz = (2 * m_info.tiling.r_h + 1) * (2 * m_info.tiling.r_w + 1) * 3;
    std::vector<Eigen::VectorXd> blendings(m_info.nds.size(), Eigen::VectorXd(sz));
    for (size_t s = 0; s < m_info.nds.size(); ++s) {
        size_t count = 0;
        for (int k = i - m_info.tiling.r_h; k < (int) (i + m_info.tiling.r_h + 1); ++k) {
            for (int m = j - m_info.tiling.r_w; m < (int) (j + m_info.tiling.r_w + 1); ++m) {
                const size_t k_ = k >= 0 ? k % m_info.shape.h : m_info.shape.h + k;
                const size_t m_ = m >= 0 ? m % m_info.shape.w : m_info.shape.w + m;
                for (size_t c = 0; c < 3; ++c) {
                    blendings[s][count++] = m_info.blending_coefs[k_][m_][s]; 
                }
            }    
        }
        assert(count == sz);
    }
    Eigen::VectorXd u = Eigen::VectorXd::Constant(sz, 0.0);
    Eigen::MatrixXd sigma = Eigen::MatrixXd::Constant(sz, sz, 0.0);
    for (size_t s = 0; s < m_info.nds.size(); ++s) {
        u += blendings[s].asDiagonal() * m_info.nds[s].u;
        sigma += blendings[s].asDiagonal() * m_info.nds[s].s * blendings[s].asDiagonal();
    }
    return {.u = u, .s = sigma};
}

Eigen::VectorXd NDImageDrawer::SamplePixels() const {
    // GibbsSamplerND sampler({.nd = m_nd, .iter = 10});
    // return sampler.Sample();
    return {};
}

uint8_t NDImageDrawer::ToImageColour(double v) const {
    return static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
}