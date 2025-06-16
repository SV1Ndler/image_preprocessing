#ifndef IMAGE_PREPROCESSING_PP_PIXEL_HPP_
#define IMAGE_PREPROCESSING_PP_PIXEL_HPP_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <inttypes.h>
#include <array>

namespace pp {

struct PixelHSI;
struct PixelHSV;
struct PixelRGB;
struct PixelRGBRef;

struct PixelRGB {
    enum Pos {
        R = 0,
        G = 1,
        B = 2,
    };

    PixelRGB() = default;

    PixelRGB(uint8_t R, uint8_t G, uint8_t B);

    PixelRGB(PixelHSI pixel);

    PixelRGB(PixelHSV pixel);

    PixelRGB(const PixelRGB& other);

    PixelRGB(PixelRGB&& other);

    PixelRGB& operator=(const PixelRGB& other);

    PixelRGB& operator=(PixelRGB&& other);
    
    void swap(PixelRGB& other);

    uint8_t& r() { return data[Pos::R]; }
    uint8_t& g() { return data[Pos::G]; }
    uint8_t& b() { return data[Pos::B]; }

    uint8_t r() const { return data[Pos::R]; }
    uint8_t g() const { return data[Pos::G]; }
    uint8_t b() const { return data[Pos::B]; }

    // noramalized
    double rn() { return data[Pos::R] / 255.; }
    double gn() { return data[Pos::G] / 255.; }
    double bn() { return data[Pos::B] / 255.; }

    uint8_t grayscale();

    operator PixelRGBRef();

    std::array<uint8_t, 3> data;
};

struct PixelRGBRef {
    using Pos = PixelRGB::Pos;

    PixelRGBRef();

    PixelRGBRef(uint8_t* p);

    PixelRGBRef(PixelRGB& other);

    PixelRGBRef(const PixelRGBRef& other);

    PixelRGBRef(PixelRGBRef&& other);

    PixelRGBRef& operator=(const PixelRGBRef& other);

    PixelRGBRef& operator=(PixelRGBRef&& other);
    
    void swap(PixelRGBRef& other);

    uint8_t& r() { return data[Pos::R]; }
    uint8_t& g() { return data[Pos::G]; }
    uint8_t& b() { return data[Pos::B]; }

    uint8_t r() const { return data[Pos::R]; }
    uint8_t g() const { return data[Pos::G]; }
    uint8_t b() const { return data[Pos::B]; }

    // noramalized
    double rn() { return data[Pos::R] / 255.; }
    double gn() { return data[Pos::G] / 255.; }
    double bn() { return data[Pos::B] / 255.; }

    uint8_t grayscale();

    operator PixelRGB() const { return PixelRGB(r(), g(), b()); }

    uint8_t* data;
};

struct PixelHSI {
    enum Pos {
        H = 0,
        S = 1,
        I = 2,
    };

    PixelHSI(double H, double S, double I);

    PixelHSI(PixelRGB pixel);

    std::array<double, 3> data;
    double& h() { return data[Pos::H]; }
    double& s() { return data[Pos::S]; }
    double& i() { return data[Pos::I]; }
};

struct PixelHSV {
    enum Pos {
        H = 0,
        S = 1,
        V = 2,
    };

    PixelHSV(double H, double S, double V);

    PixelHSV(PixelRGB pixel);

    std::array<double, 3> data;
    double& h() { return data[Pos::H]; }
    double& s() { return data[Pos::S]; }
    double& v() { return data[Pos::V]; }
};

uint8_t& Get(uint8_t* pixel, PixelRGB::Pos pos);

} //namespace pp

#endif