#include "pp/mat/mat.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "pp/pixel/pixel.hpp"

namespace pp {

// TODO: Добавить throw(except)

Mat::Mat(std::size_t rows, std::size_t cols, const unsigned char* data): Mat{rows, cols} {
    std::memcpy(this->data, data, rows*cols*3);
}

Mat::Mat(std::size_t rows, std::size_t cols, unsigned char* data): rows(rows), cols(cols), data(data) {}

Mat::Mat()
    : rows{0}, cols{0}, data{nullptr} {}

Mat::~Mat(){
    if(data != nullptr) {
        delete [] data;
    }
}

bool Mat::operator==(const Mat& other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }
    
    if (data == other.data) {
        return true;
    }
    const size_t totalPixels = rows * cols * 3;
    return std::memcmp(data, other.data, totalPixels) == 0;
}
uint8_t* Mat::GetPtr(std::size_t row, std::size_t col) {
    return data + (cols * row + col) * 3;
}

const uint8_t* Mat::GetPtr(std::size_t row, std::size_t col) const {
    return data + (cols * row + col) * 3;
}

PixelRGB Mat::GetPixel(std::size_t row, std::size_t col) const {
    const uint8_t* p = GetPtr(row, col);
    return PixelRGB(p[PixelRGBRef::Pos::R], p[PixelRGBRef::Pos::G], p[PixelRGBRef::Pos::B]);
}

PixelRGBRef Mat::GetPixel(std::size_t row, std::size_t col) {
    uint8_t* p = GetPtr(row, col);
    return PixelRGBRef(p);
}

void Mat::SetPixel(std::size_t row, std::size_t col, const PixelRGBRef& pixel) {
    uint8_t* p = GetPtr(row, col);
    p[0] = pixel.r();
    p[1] = pixel.g();
    p[2] = pixel.b();
}

Mat Mat::CopyMakeBorder(std::size_t borderSize) {
    Mat result = CopyWithBorder(borderSize);
    result.MakeBorder(borderSize);

    return result;
}

Mat Mat::CopyWithBorder(std::size_t borderSize) {
    Mat result(rows + borderSize*2, cols + borderSize*2);
    result.borderSize = borderSize;

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            auto p = GetPixel(i, j);
            result.SetPixel(i + borderSize, j + borderSize, p);
        }
    }

    return result;
}

Mat Mat::ResetBorder() {
    if(borderSize == 0) {
        return *this;
    }

    Mat result(rows - borderSize*2, cols - borderSize*2);

    for (std::size_t i = 0; i < result.rows; ++i) {
        for (std::size_t j = 0; j < result.cols; ++j) {
            auto p = GetPixel(i + borderSize, j + borderSize);
            result.SetPixel(i, j, p);
        }
    }

    return result;
}

void Mat::MakeBorder(std::size_t borderSize) {
    // Заполняем зеркально края
    // Верх и них
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            auto pixUpper = GetPixel(i, j);
            SetPixel(borderSize - 1 - i, j + borderSize, pixUpper);

            auto pixBottom = GetPixel(rows - 1 - i, j);
            SetPixel(borderSize + rows + i, j + borderSize, pixBottom);
        }
    }

    // Слева и справа
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixLeft = GetPixel(i, j);
            SetPixel(i + borderSize, borderSize - 1 - j, pixLeft);

            auto pixRight = GetPixel(i, cols - 1 - j);
            SetPixel(i + borderSize, borderSize + cols + j, pixRight);
        }
    }

    // Углы
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixTopLeft = GetPixel(borderSize - i - 1, borderSize - j - 1);
            SetPixel(i, j, pixTopLeft);

            auto pixTopRight = GetPixel(borderSize - i - 1, cols - 1 - j);
            SetPixel(i, borderSize + cols + j, pixTopRight);

            auto pixBottomLeft = GetPixel(rows - 1 - i, borderSize - j - 1);
            SetPixel(borderSize + rows + i, j, pixBottomLeft);

            auto pixBottomRight = GetPixel(rows - 1 - i, cols - 1 - j);
            SetPixel(borderSize + rows + i, borderSize + cols + j, pixBottomRight);
        }
    }
}

void Mat::MirrorTopEdge(std::size_t borderSize) {
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < cols - 2 * borderSize; ++j) {
            auto pixel = GetPixel(borderSize + i, borderSize + j);
            SetPixel(borderSize - 1 - i, borderSize + j, pixel);
        }
    }
}

void Mat::MirrorBottomEdge(std::size_t borderSize) {
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < cols - 2 * borderSize; ++j) {
            auto pixel = GetPixel(rows - borderSize - 1 - i, borderSize + j);
            SetPixel(rows - borderSize + i, borderSize + j, pixel);
        }
    }
}

void Mat::MirrorLeftEdge(std::size_t borderSize) {
    for (std::size_t i = 0; i < rows - 2 * borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixel = GetPixel(borderSize + i, borderSize + j);
            SetPixel(borderSize + i, borderSize - 1 - j, pixel);
        }
    }
}

void Mat::MirrorRightEdge(std::size_t borderSize) {
    for (std::size_t i = 0; i < rows - 2 * borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixel = GetPixel(borderSize + i, cols - borderSize - 1 - j);
            SetPixel(borderSize + i, cols - borderSize + j, pixel);
        }
    }
}

void Mat::MirrorTopLeftCorner(std::size_t borderSize) {
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixel = GetPixel(borderSize + i, borderSize + j);
            SetPixel(borderSize - 1 - i, borderSize - 1 - j, pixel);
        }
    }
}


void Mat::MirrorTopRightCorner(std::size_t borderSize) {
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixel = GetPixel(borderSize + i, cols - borderSize - 1 - j);
            SetPixel(borderSize - 1 - i, cols - borderSize + j, pixel);
        }
    }
}

void Mat::MirrorBottomLeftCorner(std::size_t borderSize) {
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixel = GetPixel(rows - borderSize - 1 - i, borderSize + j);
            SetPixel(rows - borderSize + i, borderSize - 1 - j, pixel);
        }
    }
}

void Mat::MirrorBottomRightCorner(std::size_t borderSize) {
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < borderSize; ++j) {
            auto pixel = GetPixel(rows - borderSize - 1 - i, cols - borderSize - 1 - j);
            SetPixel(rows - borderSize + i, cols - borderSize + j, pixel);
        }
    }
}

void Mat::MirrorEdges(std::size_t borderSize) {
    MirrorTopEdge(borderSize);
    MirrorBottomEdge(borderSize);
    MirrorLeftEdge(borderSize);
    MirrorRightEdge(borderSize);
}

void Mat::MirrorCorners(std::size_t borderSize) {
    MirrorTopLeftCorner(borderSize);
    MirrorTopRightCorner(borderSize);
    MirrorBottomLeftCorner(borderSize);
    MirrorBottomRightCorner(borderSize);
}

void Mat::MakeMirrorBorder(std::size_t borderSize) {
    MirrorEdges(borderSize);
    MirrorCorners(borderSize);
}

uint8_t* ROI::GetPtr(std::size_t row, std::size_t col) {
    return mat->GetPtr(row + rect.y, col + rect.x);
}

const uint8_t* ROI::GetPtr(std::size_t row, std::size_t col) const {
    return mat->GetPtr(row + rect.y, col + rect.x);
}

PixelRGBRef ROI::GetPixel(std::size_t row, std::size_t col) {
    uint8_t* p = GetPtr(row, col);
    return PixelRGBRef { p };
}

PixelRGB ROI::GetPixel(std::size_t row, std::size_t col) const {
    const uint8_t* p = GetPtr(row, col);
    return PixelRGB { p[PixelRGBRef::Pos::R], p[PixelRGBRef::Pos::G], p[PixelRGBRef::Pos::B] };
}

void ROI::SetPixel(std::size_t row, std::size_t col, const PixelRGBRef& pixel) {
    uint8_t* p = GetPtr(row, col);
    p[0] = pixel.r();
    p[1] = pixel.g();
    p[2] = pixel.b();
}

} // namespace pp