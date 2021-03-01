#include <NDImageDrawer.h>
#include <GibbsSamplerND.h>
#include <NDSampler.h>
#include <Logger.h>

NDImageDrawer::NDImageDrawer(const Info& info)
: m_info(info) {}

cv::Mat NDImageDrawer::Draw(size_t iter) const {
    cv::Mat img = GenerateRandomFromDist();
    cv::imshow("result", img);
    cv::waitKey();
    for (size_t i = 0; i < iter; ++i) {
        std::cout << i << std::endl;
        UpdateWithGibbsSampling(img);
        cv::imshow("result", img);
        cv::waitKey();
    }
    return img;
}

cv::Mat NDImageDrawer::GenerateRandomFromDist() const {
    const auto tiling_h = m_info.tiling.r_h * 2 + 1;
    const auto tiling_w = m_info.tiling.r_w * 2 + 1;
    cv::Mat img(m_info.shape.h, m_info.shape.w, CV_8UC3);
    NDSampler sampler(m_info.nd);
    for (size_t i = 0; i < m_info.shape.h / tiling_h; ++i) {
        for (size_t j = 0; j < m_info.shape.w / tiling_w; ++j) {
            const auto vals = sampler.Sample();
            size_t count = 0;
            for (size_t k = 0; k < tiling_h; ++k) {
                for (size_t m = 0; m < tiling_w; ++m) {
                    auto& img_val = img.at<cv::Vec3b>(static_cast<int>(i * tiling_h + k), 
                                                      static_cast<int>(j * tiling_w + m));
                    img_val[0] = ToImageColour(vals[count++]);
                    img_val[1] = ToImageColour(vals[count++]);
                    img_val[2] = ToImageColour(vals[count++]);
                }
            }
        }
    }
    return img;
}

void NDImageDrawer::UpdateWithGibbsSampling(cv::Mat& img) const {
    GibbsSamplerND sampler({.nd = m_info.nd, .iter = 0});
    // size_t i = m_info.tiling.r_h;
    // size_t j = m_info.tiling.r_h;
    // const auto vals = GetNeighbors(img, i, j);
    // std::cout << "[";
    // for (const auto& v : vals) {
    //     std::cout << v << ", ";
    // }
    // std::cout << "]" << std::endl;
    
    const auto pos = (2 * m_info.tiling.r_h + 1) * (2 * m_info.tiling.r_w + 1) / 2;
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
    for (size_t i = m_info.tiling.r_h; i < m_info.shape.h - m_info.tiling.r_h; ++i) {
        for (size_t j = m_info.tiling.r_w; j < m_info.shape.w - m_info.tiling.r_w; ++j) {
            // auto& img_val = img.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
            // std::cout << "BEFORE: " <<  img_val << std::endl;
            const auto vals = GetNeighbors(img, i, j);
            const auto new_pixel = sampler.SampleWithVals(vals, pos);
            auto& img_val = img.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
            for (size_t c = 0; c < 3; ++c) {
                // std::cout << "before: " << (int) img_val[0] << " " << (int) img_val[1] << " "  << (int) img_val[2] << std::endl;
                img_val[c] = ToImageColour(new_pixel[c]); 
                // std::cout << "after: " << (int) img_val[0] << " "  << (int) img_val[1] << " "  << (int) img_val[2] << std::endl;
            }
            // std::cout << "AFTER: " <<  img_val << std::endl;
        }
    }
}

Eigen::VectorXd NDImageDrawer::GetNeighbors(const cv::Mat& img, size_t i, size_t j) const {
    size_t count = 0;
    auto nd_vec = Eigen::VectorXd((2 * m_info.tiling.r_h + 1) * (2 * m_info.tiling.r_w + 1) * 3);
    for (size_t k = i - m_info.tiling.r_h; k < i + m_info.tiling.r_h + 1; ++k) {
        for (size_t m = j - m_info.tiling.r_w; m < j + m_info.tiling.r_w + 1; ++m) {
            const auto& val = img.at<cv::Vec3b>(static_cast<int>(k), static_cast<int>(m));
            nd_vec[count++] = val[0];
            nd_vec[count++] = val[1];
            nd_vec[count++] = val[2];
        }
    }
    return nd_vec;
}

Eigen::VectorXd NDImageDrawer::SamplePixels() const {
    // GibbsSamplerND sampler({.nd = m_nd, .iter = 10});
    // return sampler.Sample();
    return {};
}

uint8_t NDImageDrawer::ToImageColour(double v) const {
    return static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
}