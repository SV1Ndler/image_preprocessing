#ifndef IMAGE_PREPROCESSING_PP_TRANSFORMATION_HPP_
#define IMAGE_PREPROCESSING_PP_TRANSFORMATION_HPP_

#include "pp/mat/mat.hpp"

#include <vector>

namespace pp {



class MeanFilterProc {
public:
    MeanFilterProc(std::size_t kernelSize): kernelSize(kernelSize) {}
    void operator()(ROI& roi, PixelRGBRef& dst);

    std::size_t kernelSize;
};

class MedianFilterProc {
public:
    MedianFilterProc(std::size_t kernelSize): kernelSize(kernelSize) {}
    void operator()(ROI& roi, PixelRGBRef& dst);

    std::size_t kernelSize;
};

class ThresholdFilterProc {
public:
    ThresholdFilterProc(uint8_t thresholdValue): thresholdValue(thresholdValue) {}
    void operator()(ROI& roi, PixelRGBRef& dst);

    uint8_t thresholdValue;
    const uint8_t kernelSize = 1;
};

class SegmentationFilterProc {
public:
    enum SegmentationFiltersParam {
        kEachChannelSeparately,
        kGrayScale,
        kMaxGradient,
    };

    SegmentationFilterProc(const std::vector<int32_t>& kernelX, const std::vector<int32_t>& kernelY, std::size_t kernelSize, SegmentationFiltersParam param = kMaxGradient);

    void operator()(ROI& roi, PixelRGBRef& dst);

    std::size_t kernelSize;
private:
    std::vector<int32_t> kernelX_;
    std::vector<int32_t> kernelY_; 

    using FunProc = void(SegmentationFilterProc::*)(ROI&, PixelRGBRef&);

    FunProc proc_;

    void ProcEachChannelSeparately(ROI& roi, PixelRGBRef& dst);
    void ProcGrayScale(ROI& roi, PixelRGBRef& dst);
    void ProcMaxGradient(ROI& roi, PixelRGBRef& dst);
};

class SobelFilterProc: public SegmentationFilterProc {
public:
    SobelFilterProc()
    : SegmentationFilterProc(
        {
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1,
        },
        {
            -1, -2, -1,
             0,  0,  0,
             1,  2,  1,
        }, 
        3
    ) {}
};

class PrewittFilterProc: public SegmentationFilterProc {
public:
    PrewittFilterProc()
    : SegmentationFilterProc(
        {
            -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1,
        },
        {
            -1, -1, -1,
             0,  0,  0,
             1,  1,  1,
        }, 
        3
    ) {}
};

template<class Processor>
void DoFilter(Mat& src, Mat& dst, Processor proc) {
    const std::size_t kernelSize = proc.kernelSize;
    int half_k = kernelSize / 2;

    for (std::size_t row = 0; row < src.rows - kernelSize + 1; ++row) {
        for (std::size_t col = 0; col < src.cols - kernelSize + 1; ++col) {
            Rect rect(col, row, kernelSize, kernelSize);
            ROI roi(src, rect);
            PixelRGBRef pixel = dst.GetPixel(row + half_k, col + half_k);

            proc(roi, pixel); // <- операция над изображением
        }
    }
}

void InitImg(Mat& src);

// template<class Processor>
// void DoFilter(ROI& src, ROI& dst, Processor proc) {
//     const std::size_t kernelSize = proc.kernelSize;
//     int half_k = kernelSize / 2;

//     for (std::size_t row = 0; row < src.rows - kernelSize + 1; ++row) {
//         for (std::size_t col = 0; col < src.cols - kernelSize + 1; ++col) {
//             Rect rect(col, row, kernelSize, kernelSize);
//             ROI roi(src, rect);
//             PixelRGBRef pixel = dst.GetPixel(row + half_k, col + half_k);

//             proc(roi, pixel); // <- операция над изображением
//         }
//     }
// }

} // namespace pp

#endif