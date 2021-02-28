#include <GibbsSamplerND.h>
#include <ImageAsND.h>
#include <NDImageDrawer.h>
#include <Logger.h>

int main() {
    const auto image = cv::imread("../images/moss.jpg", cv::IMREAD_COLOR);
    // std::cout << image.rows << image.cols << std::endl;
    // 
    cv::imshow("result",image);
    cv::waitKey();
    size_t h = 35, w = 35;
    cv::Mat cropped = cv::Mat(h, w, CV_8UC3);
    for (size_t i = 0; i < h; ++i) {
        for (size_t j = 0; j < w; ++j) {
            cropped.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i + 100, j + 100);
        }
    
    }
    ImageAsND img_to_nd(cropped, {.w = 5, .h = 5});
    auto nd = img_to_nd.ComputeND();
    GibbsSamplerND sampler({.nd = nd, .iter = 10});
    const auto x = sampler.Sample();
    std::cout << x << std::endl;
    // nd.s = nd.s / 10;
    // GibbsSamplerND sampler({.nd = nd, .iter = 100000});
    // for (size_t i = 0; i < 100; ++i) {
    //     LOG(sampler.Sample());
    //     LOG_ENDL;
    // }
    // 
    NDImageDrawer drawer({.nd = nd, .w = 50, .h = 50, .r_w = 5, .r_h = 5});
    const auto sampled_img = drawer.Draw();
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    cv::imwrite("sampled_moss.png", sampled_img, compression_params);
    std::vector<int> compression_params_2;
    compression_params_2.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params_2.push_back(0);
    cv::imwrite("sampled_grass_0.png", sampled_img, compression_params_2);
    

    return 0;
}