#include <bits/types/struct_timeval.h>
#include <cstddef>
#include <cstdio>
#include <omp.h>
#include <random>
#include <sys/select.h>
#include <sys/time.h>

#include "pp/mat/mat.hpp"
#include "pp/pixel/pixel.hpp"
#include "pp/transformation/transformation.hpp"

using pp::Mat;

void MakeMirrorBorder(Mat& img, std::size_t borderSize);

void InitImg(Mat& src);

template<class Processor>
void DoFilter(pp::Mat& src, pp::Mat& dst, Processor proc) {
    const std::size_t kernelSize = proc.kernelSize;
    int half_k = kernelSize / 2;

    // std::size_t rowStart = (threadId != 0)? 

    #pragma omp for schedule(static, 64)
    for (std::size_t row = 0; row < src.rows - kernelSize + 1; ++row) {
        for (std::size_t col = 0; col < src.cols - kernelSize + 1; ++col) {
            pp::Rect rect(col, row, kernelSize, kernelSize);
            pp::ROI roi(src, rect);
            pp::PixelRGBRef pixel = dst.GetPixel(row + half_k, col + half_k);

            proc(roi, pixel); // <- операция над изображением
        }
    }
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double) t.tv_usec * 1E-6;

}

int main() {
    const std::size_t kRows = 5000;//1080;
    const std::size_t kColls = 5000;//2048;
    const std::size_t kBorderSize = 3;

    // const int N = 1;
    omp_get_thread_num();
    
    pp::Mat img;

    // double sum_t = 0;
    double t;


    pp::Mat correctImg;


    {
        pp::Mat img1(kRows, kColls);
        img1 = img1.CopyWithBorder(kBorderSize);
        pp::Mat img2(img1.rows, img1.cols, img1.borderSize);
        
        t = wtime();
    
        pp::InitImg(img1);
        pp::Mat* img = &img1;
        pp::Mat* imgNew = &img2;

        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::MedianFilterProc(7));
        std::swap(img, imgNew);
        
        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::MeanFilterProc(7));
        std::swap(img, imgNew);

        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::SobelFilterProc());
        std::swap(img, imgNew);

        img->MakeMirrorBorder(kBorderSize);
        pp::DoFilter(*img, *imgNew, pp::ThresholdFilterProc(20));
        std::swap(img, imgNew);

       
        correctImg.swap(*img);

        t = wtime() - t;
        printf("(S)Elapsed time (sec.): %.12f\n", t);
    }

    

    pp::Mat img1(kRows, kColls);
    img1 = img1.CopyWithBorder(kBorderSize);
    pp::Mat img2(img1.rows, img1.cols, img1.borderSize);

    bool result;
    #pragma omp parallel shared(img1, img2)
    {
        #pragma omp barrier
        #pragma omp master
        {
            t = wtime();
        }

        ::InitImg(img1);
        pp::Mat* img = &img1;
        pp::Mat* imgNew = &img2;

        MakeMirrorBorder(*img, kBorderSize); 
        ::DoFilter(*img, *imgNew, pp::MedianFilterProc(7));
        std::swap(img, imgNew); 
        

        MakeMirrorBorder(*img, kBorderSize); 
        ::DoFilter(*img, *imgNew, pp::MeanFilterProc(7));
        std::swap(img, imgNew);

        MakeMirrorBorder(*img, kBorderSize); 
        ::DoFilter(*img, *imgNew, pp::SobelFilterProc());
        std::swap(img, imgNew);


        MakeMirrorBorder(*img, kBorderSize); 
        ::DoFilter(*img, *imgNew, pp::ThresholdFilterProc(20));
        std::swap(img, imgNew);

        

        #pragma omp barrier
        #pragma omp master
        {
            t = wtime() - t;
            result = (*img == correctImg);
        }
    }
    
    
    printf("Result: %d\n", result);
    printf("(P)Elapsed time (sec.): %.12f\n", t);



    return 0;
}

void InitImg(Mat& src) {
    #pragma omp for schedule(static, 64)
    for (std::size_t row = src.borderSize; row < src.rows - src.borderSize; ++row) {
        for (std::size_t col = src.borderSize; col < src.cols - src.borderSize; ++col) {
            auto pixel = src.GetPixel(row, col);
            pixel.r() = (653 + row * src.cols + col) % 256;
            pixel.g() = (1754 + row * src.cols + col) % 256;
            pixel.b() = (1999 + row * src.cols + col) % 256;
        }
    }
}

void MakeMirrorBorder(Mat& img, std::size_t borderSize) {
    #pragma omp for collapse(2) nowait
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < img.cols - 2 * borderSize; ++j) {
            auto topPixel = img.GetPixel(borderSize + i, borderSize + j);
            img.SetPixel(borderSize - 1 - i, borderSize + j, topPixel);

            auto bottomPixel = img.GetPixel(img.rows - borderSize - 1 - i, borderSize + j);
            img.SetPixel(img.rows - borderSize + i, borderSize + j, bottomPixel);
        }
    }

    #pragma omp for nowait
    for (std::size_t i = 0; i < img.rows - 2 * borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto leftPixel = img.GetPixel(borderSize + i, borderSize + j);
            img.SetPixel(borderSize + i, borderSize - 1 - j, leftPixel);

            auto rightPixel = img.GetPixel(borderSize + i, img.cols - borderSize - 1 - j);
            img.SetPixel(borderSize + i, img.cols - borderSize + j, rightPixel);
        }
    }

    #pragma omp for collapse(2)
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto tl = img.GetPixel(borderSize + i, borderSize + j);
            img.SetPixel(borderSize - 1 - i, borderSize - 1 - j, tl);

            auto tr = img.GetPixel(borderSize + i, img.cols - borderSize - 1 - j);
            img.SetPixel(borderSize - 1 - i, img.cols - borderSize + j, tr);

            auto bl = img.GetPixel(img.rows - borderSize - 1 - i, borderSize + j);
            img.SetPixel(img.rows - borderSize + i, borderSize - 1 - j, bl);

            auto br = img.GetPixel(img.rows - borderSize - 1 - i, img.cols - borderSize - 1 - j);
            img.SetPixel(img.rows - borderSize + i, img.cols - borderSize + j, br);
        }
    }
}