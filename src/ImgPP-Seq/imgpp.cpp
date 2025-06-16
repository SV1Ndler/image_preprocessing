#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>

#include "configuration/parser/parser.hpp"
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
    auto config = configuration::parse("config.json");
    config.log();

    omp_set_num_threads(config.numThreads);

    pp::Mat img;

    double sum_t = 0;
    double t;

    int N = 1;

    for(auto i = 0; i < 1; ++i) {
        cv::Mat img__ = cv::imread(config.in);
        img = pp::Mat(img__.rows, img__.cols, img__.data);

        t = omp_get_wtime();

        for(auto& filter: config.filters) {
            auto tmp = filter->apply(img);
            img.swap(tmp);
        }

        sum_t += omp_get_wtime() - t;
    }
    
    
    // printf("Result: %d\n", img0 == img1);
    printf("Elapsed time (sec.): %.12f\n", sum_t / N);

    
    cv::Mat i = cv::Mat(img.rows, img.cols, CV_8UC3, img.data);
    cv::imwrite(config.out, i);

    // cv::imwrite("image01_res.jpg", i);

    return 0;
}