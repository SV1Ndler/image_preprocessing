#ifndef IMAGE_PREPROCESSING_UTILS_OPENMP_HPP_
#define IMAGE_PREPROCESSING_UTILS_OPENMP_HPP_

#include <vector>
#include "pp/mat/mat.hpp"

namespace utils::openmp {

using pp::ROI;

void MakeBorder(ROI& roi, std::size_t borderSize) {
    roi.margin = 0;
    // Заполняем зеркально края
    // Верх и них
    for (std::size_t i = 0; i < borderSize; ++i) {
        for (std::size_t j = 0; j < roi.cols; ++j) {
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

    margin = borderSize;
}


} // namespace utils::openmp

#endif