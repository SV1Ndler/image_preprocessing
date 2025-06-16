#include "pp/pixel/pixel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace pp {

namespace {
    template<typename T>
    T roundAndStaticCast(double val) {
        // val > 255?
        return static_cast<T>(std::round(val));
    }

    double degreeToRadian(double degree) {
        return M_PI * degree / 180.;
    }

    double cosDegree(double degree) {
        return std::cos(degreeToRadian(degree));
    }
}


PixelRGB::PixelRGB(uint8_t R, uint8_t G, uint8_t B): PixelRGB() {
    r() = R;
    g() = G;
    b() = B;
}

PixelRGB::PixelRGB(PixelHSI pixel): PixelRGB() {
    // Предполагается, что H от 0 до 360 градусов;
    struct {
        double H;
        uint8_t* v1;
        uint8_t* v2;
        uint8_t* v3;
    } params;

    if (pixel.h() < 120) {
        params.H = pixel.h();
        params.v1 = &b();
        params.v2 = &r();
        params.v3 = &g();
    } else if (pixel.h() < 240) {
        params.H = pixel.h() - 120;
        params.v1 = &r();
        params.v2 = &g();
        params.v3 = &b();
    } else {
        params.H = pixel.h() - 240;
        params.v1 = &g();
        params.v2 = &b();
        params.v3 = &r();
    }

    *params.v1 = roundAndStaticCast<uint8_t>(pixel.i()*(1 - pixel.s()));
        
    const double tmp =
        ( pixel.s() * cosDegree(params.H) ) /
            cosDegree(60 - params.H);
    *params.v2 = roundAndStaticCast<uint8_t>(pixel.i()*(1. + tmp));

    *params.v3 = 3.*pixel.i() - (*params.v1 + *params.v2);
}

PixelRGB::PixelRGB(PixelHSV pixel): PixelRGB() {
    // Предполагается, что H от 0 до 360 градусов;
    const double C = pixel.v() * pixel.s();
    const double X = C*( 1 - std::abs(std::fmod( pixel.h()/60., 2) - 1 ) );
    const double m = pixel.v() - C;

    double r_, g_, b_;
    if (pixel.h() < 60) {
        r_ = C;
        g_ = X;
        b_ = 0;
    } else if (pixel.h() < 120) {
        r_ = X;
        g_ = C;
        b_ = 0;
    } else if (pixel.h() < 180) {
        r_ = 0;
        g_ = C;
        b_ = X;
    } else if (pixel.h() < 240) {
        r_ = 0;
        g_ = X;
        b_ = C;
    } else if (pixel.h() < 300) {
        r_ = X;
        g_ = 0;
        b_ = C;
    } else {
        r_ = C;
        g_ = 0;
        b_ = X;
    }

    r() = roundAndStaticCast<uint8_t>((r_+m)*255);
    g() = roundAndStaticCast<uint8_t>((g_+m)*255);
    b() = roundAndStaticCast<uint8_t>((b_+m)*255);
}

PixelRGB::PixelRGB(const PixelRGB& other): PixelRGB() {
    r() = other.r();
    g() = other.g();
    b() = other.b();
}

PixelRGB::PixelRGB(PixelRGB&& other): PixelRGB() {
    swap(other);
}

PixelRGB& PixelRGB::operator=(const PixelRGB& other) {
    PixelRGB(other).swap(*this);

    return *this;
}

PixelRGB& PixelRGB::operator=(PixelRGB&& other) {
    PixelRGB(std::move(other)).swap(*this);

    return *this;
}
    
void PixelRGB::swap(PixelRGB& other) {
    using std::swap;
    swap(data, other.data);
}

uint8_t PixelRGB::grayscale() {
    return roundAndStaticCast<uint8_t>((0.2989*r())+(0.5870*g())+(0.1141*b()));
}

PixelRGB::operator PixelRGBRef() {
    return PixelRGBRef(*this);
}


PixelRGBRef::PixelRGBRef(uint8_t* p): data{p} {}

PixelRGBRef::PixelRGBRef(PixelRGB& other): data{ other.data.data() } {}

PixelRGBRef::PixelRGBRef(const PixelRGBRef& other): PixelRGBRef(other.data) {}

PixelRGBRef::PixelRGBRef(PixelRGBRef&& other): PixelRGBRef(other.data) {}

PixelRGBRef& PixelRGBRef::operator=(const PixelRGBRef& other) {
    PixelRGBRef(other).swap(*this);

    return *this;
}

PixelRGBRef& PixelRGBRef::operator=(PixelRGBRef&& other) {
    PixelRGBRef(std::move(other)).swap(*this);

    return *this;
}
    
void PixelRGBRef::swap(PixelRGBRef& other) {
    using std::swap;
    swap(data, other.data);
}

uint8_t PixelRGBRef::grayscale() {
    return roundAndStaticCast<uint8_t>((0.2989*r())+(0.5870*g())+(0.1141*b()));
}



PixelHSI::PixelHSI(double H, double S, double I) {
    h() = H;
    s() = S;
    i() = I;
}

PixelHSI::PixelHSI(PixelRGB pixel) {
    double tetaUpper =
      (1 / 2.) * ((pixel.rn() - pixel.gn()) + (pixel.rn() - pixel.bn()));
  double tetaLower =
      std::pow(std::pow(pixel.rn() - pixel.gn(), 2) +
                   (pixel.rn() - pixel.bn()) * (pixel.gn() - pixel.bn()),
               0.5) +
      1e-6;
  double teta = std::acos(tetaUpper / tetaLower);

  if (pixel.b() <= pixel.g()) {
    h() = teta;
  } else {
    h() = 360. - teta;
  }

  auto sum = 0 + pixel.rn() + pixel.gn() + pixel.bn();
  auto min = 0 + std::min({ pixel.rn(), pixel.gn(), pixel.bn() });
  s() = 1. - (3.*min)/sum;

  i() = 1./(3.*sum);
}

PixelHSV::PixelHSV(double H, double S, double V) {
    h() = H;
    s() = S;
    v() = V;
}

PixelHSV::PixelHSV(PixelRGB pixel) {
    const double Cmax = std::max({ pixel.rn(), pixel.gn(), pixel.bn() });
    const double Cmin = std::min({ pixel.rn(), pixel.gn(), pixel.bn() });
    const double delta = Cmax - Cmin;

    if (0 == delta) {
        h() = 0;
    } else if (Cmax == pixel.rn()) {
        const double tmp = (pixel.gn() - pixel.bn()) / delta;
        h() = 60.*(std::fmod(tmp, 6));
    } else if (Cmax == pixel.gn()) {
        const double tmp = (pixel.bn() - pixel.rn()) / delta;
        h() = 60.*(tmp + 2);
    } else {
        const double tmp = (pixel.rn() - pixel.gn()) / delta;
        h() = 60.*(tmp + 4);
    }

    if (Cmax == 0) {
        s() = 0;
    } else {
        s() = delta / Cmax;
    }

    v() = Cmax;
}

uint8_t& Get(uint8_t* pixel, PixelRGB::Pos pos) {
    return pixel[pos];
}

}