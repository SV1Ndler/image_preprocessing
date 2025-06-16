#ifndef IMAGE_PREPROCESSING_PP_MAT_HPP_
#define IMAGE_PREPROCESSING_PP_MAT_HPP_

#include "pp/pixel/pixel.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <inttypes.h>
#include <utility>

namespace pp {

template <std::size_t N>
struct Vec {
    uint8_t operator[](std::size_t idx);
private:
    uint8_t data_[N];
};

struct Rect {
    Rect(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
     : x{x}, y{y}, width{width}, height{height} {}

    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t width = 0;
    std::size_t height = 0;
};

class Mat {
public:
    Mat();

    ~Mat();

    Mat(const Mat& other): Mat(other.rows, other.cols, (const unsigned char*) other.data) { borderSize = other.borderSize; }
    Mat(Mat&& other): Mat() {
        other.swap(*this);
    }
    
    Mat& operator=(const Mat& other) {
        Mat(other).swap(*this);

        return *this;
    }

    Mat& operator=(Mat&& other) {
        Mat(std::move(other)).swap(*this);

        return *this;
    }

    void swap(Mat& other) {
        using std::swap;

        swap(rows, other.rows);
        swap(cols, other.cols);
        swap(data, other.data);
        swap(borderSize, other.borderSize);
    }

    bool operator==(const Mat& other) const;

    // Оператор неравенства
    bool operator!=(const Mat& other) const {
        return !(*this == other);
    }

    Mat(std::size_t rows, std::size_t cols)
    : rows{rows}, cols{cols}, data{new uint8_t[rows*cols*3]} {}

    Mat(std::size_t rows, std::size_t cols, std::size_t borderSize): Mat{rows, cols} {
        this->borderSize = borderSize;
    }

    Mat(std::size_t rows, std::size_t cols, const unsigned char* data);

    Mat(std::size_t rows, std::size_t cols, unsigned char* data);


    PixelRGB GetPixel(std::size_t row, std::size_t col) const;
    PixelRGBRef GetPixel(std::size_t row, std::size_t col);
    void SetPixel(std::size_t row, std::size_t col, const PixelRGBRef& pixel); 

    uint8_t* GetPtr(std::size_t row, std::size_t col);
    const uint8_t* GetPtr(std::size_t row, std::size_t col) const;

    Mat CopyMakeBorder(std::size_t borderSize);
    Mat CopyWithBorder(std::size_t borderSize);
    Mat ResetBorder();

    void MakeBorder(std::size_t borderSize);

    void MirrorTopEdge(std::size_t borderSize);
    void MirrorBottomEdge(std::size_t borderSize);

    void MirrorLeftEdge(std::size_t borderSize);
    void MirrorRightEdge(std::size_t borderSize);

    void MirrorTopLeftCorner(std::size_t borderSize);
    void MirrorTopRightCorner(std::size_t borderSize);
    void MirrorBottomLeftCorner(std::size_t borderSize);
    void MirrorBottomRightCorner(std::size_t borderSize);

    void MirrorEdges(std::size_t borderSize);
    void MirrorCorners(std::size_t borderSize);
    void MakeMirrorBorder(std::size_t borderSize);

    std::size_t rows;
    std::size_t cols;
    uint8_t* data;
    std::size_t borderSize = 0;
};

class ROI {
public:
    class Iterator {
    public:
        friend class ROI;

        uint8_t* operator*() { return roi->GetPtr(y, x);}
        Iterator& operator++() {
            ++x;

            if(x >= roi->cols) {
                x = 0;
                ++y;
            }

            return *this;
        }
        Iterator& operator--() {
            if(x == 0) {
                x = roi->cols - 1;
                --y;
            } else {
                --x;
            }

            return *this;
        }
        
        
        bool operator==(const Iterator& it) const { return x == it.x && y == it.y; }
        bool operator!=(const Iterator& it) const { return !(*this == it); }
    private:
        Iterator(ROI* roi, std::size_t x, std::size_t y): x{x}, y{y}, roi{roi} {}

        std::size_t x = 0;
        std::size_t y = 0;
        ROI* roi;
    };

    friend class Iterator;

    ROI(Mat& mat, Rect& rect)
     :  rows{rect.height}, cols{rect.width}, mat{&mat}, rect{rect} {}

    ROI(ROI& roi, Rect& rect)
     :  rows{rect.height}, cols{rect.width}, mat{roi.mat}, rect{rect} {
        this->rect.x += roi.rect.x;
        this->rect.y += roi.rect.y;
     }

    ROI(Mat& mat)
     :  rows{mat.rows}, cols{mat.cols}, mat{&mat}, rect{Rect(0, 0, mat.cols, mat.rows)} {}
    
    PixelRGBRef GetPixel(std::size_t row, std::size_t col);
    PixelRGB GetPixel(std::size_t row, std::size_t col) const;
    void SetPixel(std::size_t row, std::size_t col, const PixelRGBRef& pixel);

    uint8_t* GetPtr(std::size_t row, std::size_t col);
    const uint8_t* GetPtr(std::size_t row, std::size_t col) const;

    Iterator begin() { return Iterator(this, 0, 0); }
    Iterator end() { return Iterator(this, 0, rows); }

    std::size_t rows;
    std::size_t cols;
private:
    Mat* mat;
    Rect rect;
};

} // namespace pp

#endif

// template <std::size_t N>
// Mat smoothing(std::size_t rows, std::size_t cols, const Mat& img) {
    
// }

// Mat MeanFilter(const Mat& img, std::size_t kernel_size) {
//     Mat result(img.rows, img.cols);

//     int half_k = kernel_size / 2;
//     int area = kernel_size * kernel_size;

//     ????????????????????????????????????????
//     for (std::size_t row = half_k; row < img.rows - half_k; ++row) {
//         for (std::size_t col = half_k; col < img.cols - half_k; ++col) {
//             std::array<uint32_t, 3> sum;

//             // Sum pixels in kernel window
//             for (int kr = -half_k; kr <= half_k; ++kr) {
//                 for (int kc = -half_k; kc <= half_k; ++kc) {

//                     auto pixel = img.GetPixel(row + kr, col + kc);
//                     sum[Pos::R] += pixel[Pos::R];
//                     sum[Pos::G] += pixel[Pos::G];
//                     sum[Pos::B] += pixel[Pos::B];
//                 }
//             }

//             // Compute average and set pixel
//             Mat::Pixel P;
//             P[Pos::R] = static_cast<uint8_t>(sum[Pos::R] / area);
//             P[Pos::G] = static_cast<uint8_t>(sum[Pos::G] / area);
//             P[Pos::B] = static_cast<uint8_t>(sum[Pos::B] / area);
//             result.SetPixel(row, col, P);
//         }
//     }
//     return result;
// }

// Mat MedianFilter(const Mat& img, std::size_t kernel_size) {
//     Mat result(img.rows, img.cols);

//     int half_k = kernel_size / 2;
//     int area = kernel_size * kernel_size;

//     ????????????????????????????????????????
//     for (std::size_t row = half_k; row < img.rows - half_k; ++row) {
//         for (std::size_t col = half_k; col < img.cols - half_k; ++col) {
//             std::array<uint32_t, 3> sum;

//             // Sum pixels in kernel window
//             for (int kr = -half_k; kr <= half_k; ++kr) {
//                 for (int kc = -half_k; kc <= half_k; ++kc) {
//                     auto pixel = img.GetPixel(row + kr, col + kc);
//                     sum[Pos::R] += pixel[Pos::R];
//                     sum[Pos::G] += pixel[Pos::G];
//                     sum[Pos::B] += pixel[Pos::B];
//                 }
//             }

//             // Compute average and set pixel
//             Mat::Pixel P;
//             P[Pos::R] = static_cast<uint8_t>(sum[Pos::R] / area);
//             P[Pos::G] = static_cast<uint8_t>(sum[Pos::G] / area);
//             P[Pos::B] = static_cast<uint8_t>(sum[Pos::B] / area);
//             result.SetPixel(row, col, P);
//         }
//     }
//     return result;
// }


// template <std::size_t N>
// Mat filter(std::size_t rows, std::size_t cols, const Mat& img, std::array<uint8_t, N> kernel) {
    
// }

// template <std::size_t N>
// Mat correlation(std::size_t rows, std::size_t cols, const Mat& img, std::array<uint8_t, N> kernel) {
    
// }

