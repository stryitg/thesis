#include <NDImageDrawer.h>
#include <GibbsSamplerND.h>
#include <Logger.h>

NDImageDrawer::NDImageDrawer(const Options& opt) 
: m_nd(opt.nd)
, m_w(opt.w)
, m_h(opt.h)
, m_r_w(opt.r_w)
, m_r_h(opt.r_h) {}

cv::Mat NDImageDrawer::Draw() const {
    cv::Mat img(m_h, m_w, CV_8UC3);
    size_t c = 0;
    for (size_t i = m_r_h; i < m_h - m_r_h; i += m_r_h + 1) {
        for (size_t j = m_r_w; j < m_w - m_r_h; j += m_r_w + 1) {
            const auto vals = SamplePixels();
            size_t count = 0;
            for (size_t k = i - m_r_h; k < i + m_r_h + 1; ++k) {
                for (size_t m = j - m_r_w; m < j + m_r_w + 1; ++m) {
                    img.at<cv::Vec3b>(k, m)[0] = ToImageColour(vals(count++));
                    img.at<cv::Vec3b>(k, m)[1] = ToImageColour(vals(count++));
                    img.at<cv::Vec3b>(k, m)[2] = ToImageColour(vals(count++));
                }
            }
            if (c % 10 == 0) {
                LOG("Written " << c);
            }
                
            ++c;
        }
    }
    return img;
}

Eigen::VectorXd NDImageDrawer::SamplePixels() const {
    GibbsSamplerND sampler({.nd = m_nd, .iter = 10000});
    return sampler.Sample();
}

uint8_t NDImageDrawer::ToImageColour(double v) const {
    return static_cast<uint8_t>(std::max(0.0, v));
}