#pragma once
#if !defined(__MEDIUMPARAMETERS_H)
#define __MEDIUMPARAMETERS_H

#include <mitsuba/core/ray.h>
#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

struct MediumParameters {
    Spectrum albedo;
    Spectrum sigmaT;
    Float g;
    Float eta;
    MediumParameters() {}
    MediumParameters(const Spectrum &albedo, Float g, Float eta, const Spectrum &sigmaT)
        : albedo(albedo), sigmaT(sigmaT), g(g), eta(eta) {}
    inline bool isRgb() const { return (sigmaT.min() != sigmaT.max()) || (albedo.min() != albedo.max()); }
};

MTS_NAMESPACE_END

#endif