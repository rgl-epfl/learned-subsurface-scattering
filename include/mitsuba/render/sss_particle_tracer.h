#pragma once
#if !defined(__SSSPARTICLETRACER_H)
#define __SSSPARTICLETRACER_H

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <time.h>
#include <vector>

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/kdtree.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/mitsuba.h>
#include <mitsuba/render/range.h>
#include <mitsuba/render/renderjob.h>


#include <mitsuba/render/mediumparameters.h>
#include <mitsuba/render/polynomials.h>


MTS_NAMESPACE_BEGIN

class ConstraintKdTree;

struct ScatterSamplingRecord { // Record used at runtime by samplers abstracting SS scattering
    Point p;
    Vector n;
    Vector outDir;
    bool isValid;
    Spectrum throughput;
    int sampledColorChannel = -1;
};

class MTS_EXPORT_RENDER ScatterSamplingRecordArray : public Object {
public:
    ScatterSamplingRecordArray(size_t size) : data(size) {}
    std::vector<ScatterSamplingRecord> data;
};

class MTS_EXPORT_RENDER Volpath3D {

public:
    static void onbDuff(const Vector &n, Vector &b1, Vector &b2)
    {
        float sign = copysignf(1.0f, n.z);
        const float a = -1.0f / (sign + n.z);
        const float b = n.x * n.y * a;
        b1 = Vector(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
        b2 = Vector(b, sign + n.y * n.y * a, -n.y);
    }

    static Matrix3x3 azimuthSpaceTransform(const Vector &refDir, const Vector &normal) {
        // Transform: World Space to Tangent Space
        Vector s, t;
        onbDuff(normal, s, t);
        Matrix3x3 tsTransform(s.x, s.y, s.z, t.x, t.y, t.z, normal.x, normal.y, normal.z);

        // Transform: Tangent Space => (0, y, z) Space
        Vector ts_refDir = tsTransform * refDir;
        float phi = atan2(ts_refDir.x, ts_refDir.y);
        Matrix3x3 rotMat({std::cos(phi), std::sin(phi), 0.f}, {-std::sin(phi), std::cos(phi), 0.f}, {0.f, 0.f, 1.f});
        Vector newRefDir = rotMat * ts_refDir;
        // Transform: (0, y, z) => Light Space
        onbDuff(newRefDir, s, t);
        Matrix3x3 lsTransform(s.x, s.y, s.z, t.x, t.y, t.z, newRefDir.x, newRefDir.y, newRefDir.z);
        return lsTransform * rotMat * tsTransform;
    }


    static Matrix3x3 azimuthSpaceTransformNew(const Vector &light_dir, const Vector &normal) {
        Vector t1 = normalize(cross(normal, light_dir));
        Vector t2 = normalize(cross(light_dir, t1));
        if (std::abs(dot(normal, light_dir)) > 0.99999f) {
            Vector s, t;
            onbDuff(light_dir, s, t);
            Matrix3x3 lsTransform(s.x, s.y, s.z, t.x, t.y, t.z, light_dir.x, light_dir.y, light_dir.z);
            return lsTransform;
        }
        Matrix3x3 lsTransform(t1.x, t1.y, t1.z, t2.x, t2.y, t2.z, light_dir.x, light_dir.y, light_dir.z);
        return lsTransform;
    }

    static constexpr int nChooseK(int n, int k) {
        return (k == 0 || n == k) ? 1 : nChooseK(n - 1, k - 1) + nChooseK(n - 1, k);
    }

    static constexpr int nPolyCoeffs(int polyOrder) { return nChooseK(3 + polyOrder, polyOrder); }

    struct PathSampleResult {
        enum EStatus { EValid, EAbsorbed, EInvalid };
        Point pOut;
        Vector dOut, outNormal;
        Spectrum throughput;
        int bounces;
        EStatus status;
    };

    struct TrainingSample {
        Vector3f dIn, dOut, inNormal, outNormal;
        Point3f pIn, pOut;
        Spectrum throughput, albedo, sigmaT;
        size_t bounces;
        Float absorptionProb;
        Float absorptionProbVar;
        Float g;
        Float ior;
        std::vector<float> shapeCoeffs;
        std::vector<float> shCoefficients;
    };

    struct SamplingConfig {
        bool ignoreZeroScatter           = true;
        bool disableRR                   = false;
        bool importanceSamplePolynomials = false;
        int maxBounces                   = 10000;
        float polynomialStepSize         = 0.1f;
        PolyUtils::PolyFitConfig polyCfg;
    };

    static PathSampleResult samplePath(const Scene *scene, const PolyUtils::Polynomial *polynomial, Sampler *sampler, Ray ray,
                                       const MediumParameters &medium, const SamplingConfig &samplingConfig);

    static std::vector<TrainingSample>
    samplePathsBatch(const Scene *scene, const Shape *shape, const MediumParameters &medium,
                     const SamplingConfig &samplingConfig, size_t batchSize, size_t nAbsSamples, const Point3f *inPos,
                     const Vector3f *inDir, Sampler *sampler, const PolyUtils::Polynomial *polynomial = nullptr,
                     const ConstraintKdTree *kdtree = nullptr, int polyOrder = 3);

    static std::vector<TrainingSample>
    samplePaths(const Scene *scene, const Shape *shape, const std::vector<MediumParameters> &medium,
                const SamplingConfig &samplingConfig, size_t nSamples, size_t batchSize, size_t nAbsSamples,
                const Point3f *inPos, const Vector3f *inDir, Sampler *sampler, const PolyUtils::Polynomial *polynomial = nullptr,
                const ConstraintKdTree *kdtree = nullptr);

    static bool acceptPolynomial(const PolyUtils::Polynomial &polynomial, Sampler *sampler) {
        float a = 0.0f;
        for (int i = 4; i < polynomial.coeffs.size(); ++i) {
            float c = polynomial.coeffs[i];
            a += c * c;
        }
        a             = std::log(a + 1e-4f);
        float sigmoid = 1.0f / (1.0f + std::exp(-a));
        sigmoid *= sigmoid;
        sigmoid *= sigmoid;
        return sampler->next1D() <= sigmoid;
    }

    static float effectiveAlbedo(const float &albedo) {
        return -std::log(1.0f - albedo * (1.0f - std::exp(-8.0f))) / 8.0f;
    }

    static Spectrum getSigmaTp(const Spectrum &albedo, float g, const Spectrum &sigmaT) {
        Spectrum sigmaS = albedo * sigmaT;
        Spectrum sigmaA = sigmaT - sigmaS;
        return (1 - g) * sigmaS + sigmaA;
    }

    static Spectrum effectiveAlbedo(const Spectrum &albedo) {
        float r = effectiveAlbedo(albedo[0]);
        float g = effectiveAlbedo(albedo[1]);
        float b = effectiveAlbedo(albedo[2]);
        Spectrum ret;
        ret.fromLinearRGB(r, g, b);
        return ret;
    }
};

MTS_NAMESPACE_END

#endif
