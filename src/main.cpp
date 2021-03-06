#include <GibbsSamplerND.h>
#include <ImageAsND.h>
#include <NDImageDrawer.h>
#include <Logger.h>
#include <NDSampler.h>

int main() {
    const auto image = cv::imread("../images/grass.jpeg", cv::IMREAD_COLOR);
    cv::imshow("result", image);
    cv::waitKey();

    const Tiling tiling = {.r_w = 5, .r_h = 5};
    ImageAsND img_to_nd(image, tiling);
    const auto nd = img_to_nd.ComputeND();
    std::cout << nd.u << std::endl;
    
    NDImageDrawer drawer({.nd = nd, .shape = {.w = 640, .h = 360}, .tiling = tiling});
    const auto img = drawer.Draw(20);
    cv::imshow("result", img);
    cv::waitKey();
    
    return 0;
}