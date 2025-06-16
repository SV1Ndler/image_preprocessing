#include "pp/transformation/transformation.hpp"
#include "pp/pixel/pixel.hpp"

#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>

namespace pp {
namespace {

template<typename T>
T roundAndStaticCast(double val) {
    // val > 255?
    return static_cast<T>(std::round(val));
}

int32_t ApplyKernel(ROI& roi, std::vector<int32_t> kernel, PixelRGB::Pos pos) {
    int32_t result = 0;

    auto it = kernel.begin();
    for(auto p: roi) {
        result += (*it)*p[pos];
        ++it;
    }

    return result;
}

std::array<int32_t, 3> ApplyKernel(ROI& roi, std::vector<int32_t> kernel) {
    std::array<int32_t, 3> g;

    g[PixelRGB::Pos::R] = ApplyKernel(roi, kernel, PixelRGB::Pos::R);
    g[PixelRGB::Pos::G] = ApplyKernel(roi, kernel, PixelRGB::Pos::G);
    g[PixelRGB::Pos::B] = ApplyKernel(roi, kernel, PixelRGB::Pos::B);

    return g;
}

uint8_t CalculateMagnitude(int32_t gx, int32_t gy) {
    return std::min(static_cast<uint8_t>(255), roundAndStaticCast<uint8_t>(std::sqrt(gx*gx + gy*gy)));
}

} // namespace

void MeanFilterProc::operator()(ROI& roi, PixelRGBRef& dst) {
    std::array<uint32_t, 3> sum;
    sum.fill(0);
    for(auto p: roi) {
        sum[PixelRGB::Pos::R] += p[PixelRGB::Pos::R];
        sum[PixelRGB::Pos::G] += p[PixelRGB::Pos::G];
        sum[PixelRGB::Pos::B] += p[PixelRGB::Pos::B];
    }

    int area = roi.rows * roi.cols;
    dst.r() = static_cast<uint8_t>(sum[PixelRGB::Pos::R] / area);
    dst.g() = static_cast<uint8_t>(sum[PixelRGB::Pos::G] / area);
    dst.b() = static_cast<uint8_t>(sum[PixelRGB::Pos::B] / area);
}

void MedianFilterProc::operator()(ROI& roi, PixelRGBRef& dst) {
    std::vector<PixelRGBRef> pixels;        
    for(auto p: roi) {
        auto pixelRGB = PixelRGBRef(p);
        pixels.push_back(pixelRGB);
    }

    PixelRGB::Pos channel;
    auto comp = [&channel](PixelRGBRef& a, PixelRGBRef& b) {
        return a.data[channel] < b.data[channel];
    };
    
    auto m = pixels.begin() + pixels.size() / 2;

    channel = PixelRGB::Pos::R;
    std::nth_element(pixels.begin(), m, pixels.end(), comp);
    dst.r() = m->r();

    channel = PixelRGB::Pos::G;
    std::nth_element(pixels.begin(), m, pixels.end(), comp);
    dst.g() = m->g();

    channel = PixelRGB::Pos::B;
    std::nth_element(pixels.begin(), m, pixels.end(), comp);
    dst.b() = m->b();
}

void ThresholdFilterProc::operator()(ROI& roi, PixelRGBRef& dst) {
    PixelRGBRef pixel = roi.GetPixel(0, 0);
    dst.r() = (pixel.r() < thresholdValue) ? 0 : 255;
    dst.g() = (pixel.g() < thresholdValue) ? 0 : 255;
    dst.b() = (pixel.b() < thresholdValue) ? 0 : 255;
}

void SegmentationFilterProc::operator()(ROI& roi, PixelRGBRef& dst) {
    return (this->*proc_)(roi, dst);
}

SegmentationFilterProc::SegmentationFilterProc(const std::vector<int32_t>& kernelX, const std::vector<int32_t>& kernelY, std::size_t kernelSize, SegmentationFiltersParam param)
 : kernelSize(kernelSize), kernelX_ {kernelX}, kernelY_ {kernelY} {
    switch (param) {
        case kGrayScale:
            proc_ = &SegmentationFilterProc::ProcGrayScale;
            break;
        case kEachChannelSeparately:
            proc_ = &SegmentationFilterProc::ProcEachChannelSeparately;
            break;
        case kMaxGradient:
            proc_ = &SegmentationFilterProc::ProcMaxGradient;
            break;
    }
 }

void SegmentationFilterProc::ProcEachChannelSeparately(ROI& roi, PixelRGBRef& dst) {
    std::array<int32_t, 3> gx = ApplyKernel(roi, kernelX_);
    std::array<int32_t, 3> gy = ApplyKernel(roi, kernelY_);

    dst.r() = CalculateMagnitude(gx[PixelRGB::Pos::R], gy[PixelRGB::Pos::R]);
    dst.g() = CalculateMagnitude(gx[PixelRGB::Pos::G], gy[PixelRGB::Pos::G]);
    dst.b() = CalculateMagnitude(gx[PixelRGB::Pos::B], gy[PixelRGB::Pos::B]);
}

void SegmentationFilterProc::ProcGrayScale(ROI& roi, PixelRGBRef& dst) {
    int32_t gx = 0;
    int32_t gy = 0;

    auto itX = kernelX_.begin();
    auto itY = kernelY_.begin();
    for(auto p: roi) {
        auto pixelRGB = PixelRGBRef(p);
        gx += (*itX)*pixelRGB.grayscale();
        ++itX;

        gy += (*itY)*pixelRGB.grayscale();
        ++itY;
    }

    const uint8_t magnitude = CalculateMagnitude(gx, gy);
    dst.r() = magnitude;
    dst.g() = magnitude;
    dst.b() = magnitude;
}

void SegmentationFilterProc::ProcMaxGradient(ROI& roi, PixelRGBRef& dst) {
    int32_t gx = 0;
    int32_t gy = 0;

    auto itX = kernelX_.begin();
    auto itY = kernelY_.begin();
    for(auto p: roi) {
        auto pixelRGB = PixelRGBRef(p);
        gx += (*itX)*pixelRGB.grayscale();
        ++itX;

        gy += (*itY)*pixelRGB.grayscale();
        ++itY;
    }

    const uint8_t magnitude = CalculateMagnitude(gx, gy);
    dst.r() = magnitude;
    dst.g() = magnitude;
    dst.b() = magnitude;
}

void InitImg(Mat& src) {
    std::mt19937 rng;
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (std::size_t row = src.borderSize; row < (src.rows - src.borderSize); ++row) {
        for (std::size_t col = src.borderSize; col < (src.cols - src.borderSize); ++col) {
            rng.seed(653 + row * src.cols + col);
            
            auto pixel = src.GetPixel(row, col);
            pixel.r() = dist(rng);
            pixel.g() = dist(rng);
            pixel.b() = dist(rng);
        }
    }
}

} // namespace pp