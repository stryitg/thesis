#include <GibbsSamplerND.h>
#include <ImageAsND.h>
#include <NDImageDrawer.h>
#include <Logger.h>
#include <NDSampler.h>

int main() {
    const auto image = cv::imread("../images/grass_dark.jpg", cv::IMREAD_COLOR);
    // cv::imshow("result", image);
    // cv::waitKey();

    const Tiling tiling = {.r_w = 5, .r_h = 5};
    ImageAsND img_to_nd(image, tiling);
    const auto nd = img_to_nd.ComputeND();
    std::cout << "[";
    for (const auto& v : nd.u) {
        std::cout << v << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "[";
    for (size_t i = 0; i < (size_t) nd.s.rows(); ++i) {
        const auto& row =  nd.s.row(i);
        std::cout << "[";
        for (size_t j = 0; j < (size_t) row.cols(); ++j) {
            std::cout << row(j);
            if (j != (size_t) row.cols() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i != (size_t) nd.s.rows() - 1)
            std::cout << ",";
    }
    std::cout << "]" << std::endl;
    
    
    NDImageDrawer drawer({.nd = nd, .shape = {.w = 640, .h = 360}, .tiling = tiling});
    const auto img = drawer.Draw(1000);
    cv::imshow("result", img);
    cv::waitKey();
    
    
    return 0;
}