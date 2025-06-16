#ifndef IMAGE_PREPROCESSING_CONFIGURATION_FILTER_HPP_
#define IMAGE_PREPROCESSING_CONFIGURATION_FILTER_HPP_

#include "pp/mat/mat.hpp"
#include "pp/transformation/transformation.hpp"
#include <cstddef>
#include <string>

namespace configuration {

namespace {

template<typename T>
std::string ValueToString(const std::string name, const T& value, const std::string after = "") {
    return name + "=" + std::to_string(value) + after;
}

}

class ImageFilter {
public:
    virtual ~ImageFilter() = default;
    virtual pp::Mat apply(pp::Mat& img) = 0;

    virtual std::string ToString() const = 0;
};

class MeanFilter : public ImageFilter {
public:
    MeanFilter(std::size_t kernelSize = 3): proc_(kernelSize) {}

    pp::Mat apply(pp::Mat& img) final {
        pp::Mat result{img.rows, img.cols, img.borderSize};
        pp::DoFilter(img, result, proc_);

        return result;
    }

    std::string ToString() const final {
        return "MeanFilter(" + ValueToString("kernelSize", proc_.kernelSize) + ")";
    }

private:
    pp::MeanFilterProc proc_;
};


class MedianFilter : public ImageFilter {
public:
    MedianFilter(std::size_t kernelSize = 3): proc_(kernelSize) {}

    pp::Mat apply(pp::Mat& img) final {
        pp::Mat result{img.rows, img.cols, img.borderSize};
        pp::DoFilter(img, result, proc_);

        return result;
    }

    std::string ToString() const final {
        return "MedianFilter(" + ValueToString("kernelSize", proc_.kernelSize) + ")";
    }

private:
    pp::MedianFilterProc proc_;
};

class SobelFilter : public ImageFilter {
public:
    pp::Mat apply(pp::Mat& img) final {
        pp::Mat result{img.rows, img.cols, img.borderSize};
        pp::DoFilter(img, result, proc_);

        return result;
    }

    std::string ToString() const final {
        return "SobelFilter()";
    }
private:
    pp::SobelFilterProc proc_;
};

class PrewittFilter : public ImageFilter {
public:
    pp::Mat apply(pp::Mat& img) final {
        pp::Mat result{img.rows, img.cols, img.borderSize};
        pp::DoFilter(img, result, proc_);

        return result;
    }

    std::string ToString() const final {
        return "PrewittFilter()";
    }
private:
    pp::PrewittFilterProc proc_;
};

class ThresholdFilter : public ImageFilter {
public:
    ThresholdFilter(uint8_t thresholdValue)
    : proc_(thresholdValue) {}

    pp::Mat apply(pp::Mat& img) final {
        pp::Mat result{img.rows, img.cols, img.borderSize};
        pp::DoFilter(img, result, proc_);

        return result;
    }

    std::string ToString() const final {
        return "ThresholdFilter(" + ValueToString("thresholdValue", proc_.thresholdValue) + ")";
    }

private:
    pp::ThresholdFilterProc proc_;
};

}

#endif