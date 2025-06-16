#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>

// #include "configuration/parser/parser.hpp"
#include "pp/mat/mat.hpp"
#include "pp/pixel/pixel.hpp"
#include "pp/transformation/transformation.hpp"

using cv::Vec3b;

// correlation and convolution

//  {
//   "type": "Mean",
//   "kernel_size": 3
// },
// {
//     "type": "Sobel"
// },
// {
//   "type": "Threshold",
//   "threshold": 77
// }

int main() {
    // auto config = configuration::parse("config.json");
    // config.log();

    // omp_set_num_threads(config.numThreads);

    pp::Mat img;

    double sum_t = 0;
    double t;

    int N = 1;

    cv::Mat img__ = cv::imread("resources/img03n.png");
    img = pp::Mat(img__.rows, img__.cols, (const unsigned char *) img__.data).CopyWithBorder(5);//.CopyWithBorder(50);
    img.MakeMirrorBorder(5);
    pp::Mat tmp = img;

    pp::DoFilter(tmp, img, pp::MedianFilterProc(5));

    // img.MakeMirrorBorder(5);
    // pp::DoFilter(img, tmp, pp::MeanFilterProc(3));

    // tmp.MakeMirrorBorder(5);
    // pp::DoFilter(tmp, img, pp::SobelFilterProc());

    // img.MakeMirrorBorder(5);
    // pp::DoFilter(img, tmp, pp::ThresholdFilterProc(70));

    // tmp.swap(img);

    auto img1 = img.ResetBorder();
    // img.SetPixel(0, 0, img.GetPixel(0, 0));
    // img.MakeBorder(3);
    // img.MakeMirrorBorder(50);

    t = omp_get_wtime();
    sum_t = t;
    
    
    // printf("Result: %d\n", img0 == img1);
    printf("Elapsed time (sec.): %.12f\n", sum_t / N);

    
    cv::Mat i = cv::Mat(img1.rows, img1.cols, CV_8UC3, img1.data);
    cv::imwrite("image_out.png", i);

    // cv::imwrite("image01_res.jpg", i);

    return 0;
}