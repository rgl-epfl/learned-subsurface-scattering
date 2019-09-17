#if !defined(__VAEHELPER_H__)
#define __VAEHELPER_H__

#include <mitsuba/core/tls.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sss_particle_tracer.h>
#include <mitsuba/render/mediumparameters.h>
#include <mitsuba/render/polynomials.h>

#include "vaeconfig.h"

MTS_NAMESPACE_BEGIN

/**
 * Helper Class to sample using the VAE
 */
class VaeHelper : public Object {
public:
    static void sampleGaussianVector(float *data, Sampler *sampler, int nVars) {
        bool odd = nVars % 2;
        int idx  = 0;
        for (int i = 0; i < nVars / 2; ++i) {
            Point2 uv = warp::squareToStdNormal(sampler->next2D());
            data[idx] = uv.x;
            ++idx;
            data[idx] = uv.y;
            ++idx;
        }
        if (odd)
            data[idx] = warp::squareToStdNormal(sampler->next2D()).x;
    }

    static void sampleUniformVector(float *data, Sampler *sampler, int nVars) {
        for (int i = 0; i < nVars; ++i) {
            data[i] = sampler->next1D();
        }
    }

    VaeHelper() {}

    virtual bool prepare(const Scene *scene, const std::vector<Shape *> &shapes, const Spectrum &sigmaT,
                         const Spectrum &albedo, float g, float eta, const std::string &modelName,
                         const std::string &absModelName, const std::string &angularModelName,
                         const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig);
    virtual ScatterSamplingRecord sample(const Scene *scene, const Point &p, const Vector &d, const Vector &polyNormal, const Spectrum &sigmaT,
                        const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its, bool projectSamples, int channel=0) const {
                            std::cout << "NOT IMPLEMENTED\n";
                            return ScatterSamplingRecord();
                        };

    virtual void sampleBatched(const Scene *scene, const Point &p, const Vector &d, const Spectrum &sigmaT,
                        const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
                        int nSamples, bool projectSamples, std::vector<ScatterSamplingRecord> &sRec) const {
                            std::cout << "NOT IMPLEMENTED\n";
                        };
    virtual void sampleRGB(const Scene *scene, const Point &p, const Vector &d, const Spectrum &sigmaT,
                    const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
                    bool projectSamples, ScatterSamplingRecord *sRec) const {
                        std::cout << "NOT IMPLEMENTED\n";
                    };

    const VaeConfig &getConfig() const { return m_config; };

    void precomputePolynomialsImpl(const std::vector<Shape *> &shapes, const MediumParameters &medium,
                                   int channel, const PolyUtils::PolyFitConfig &pfConfig);

    void precomputePolynomials(const std::vector<Shape *> &shapes, const MediumParameters &medium,
                               const PolyUtils::PolyFitConfig &pfConfig);

    static size_t numPolynomialCoefficients(size_t deg) { return (deg + 1) * (deg + 2) * (deg + 3) / 6; }

    std::vector<float> getPolyCoeffs(const Point &p, const Vector &d, Float sigmaT_scalar, Float g,
                                     const Spectrum &albedo, const Intersection *its, bool useLightSpace,
                                     std::vector<float> &shapeCoeffs, std::vector<float> &tmpCoeffs,
                                     bool useLegendre = false, int channel = 0) const;

    template <size_t PolyOrder = 3>
    Eigen::Matrix<float, PolyUtils::nPolyCoeffs(PolyOrder), 1> getPolyCoeffsEigen(const Point &p, const Vector &d,
                                                                       const Vector &polyNormal,
                                                                       const Intersection *its, bool useLightSpace,
                                                                       bool useLegendre = false, int channel=0) const {
        if (its) {
            // const Eigen::VectorXf &c = its->polyCoeffs;
            const float *coeffs = its->polyCoeffs[channel];
            if (useLightSpace) {
                Vector s, t;
                Vector n = -d;
                Volpath3D::onbDuff(n, s, t);
                Eigen::Matrix<float, PolyUtils::nPolyCoeffs(PolyOrder), 1> shapeCoeffs = PolyUtils::rotatePolynomialEigen<PolyOrder>(coeffs, s, t, n);
                if (useLegendre)
                    PolyUtils::legendreTransform(shapeCoeffs);
                return shapeCoeffs;
            } else {
                Eigen::Matrix<float, PolyUtils::nPolyCoeffs(PolyOrder), 1> shapeCoeffs;
                for (int i = 0; i < its->nPolyCoeffs; ++i)
                    shapeCoeffs[i] = coeffs[i];
                if (useLegendre)
                    PolyUtils::legendreTransform(shapeCoeffs);
                return shapeCoeffs;
            }
        }
        return Eigen::Matrix<float, PolyUtils::nPolyCoeffs(PolyOrder), 1>::Zero();
    }
    template <size_t PolyOrder = 3>
    std::pair<Eigen::Matrix<float, PolyUtils::nPolyCoeffs(PolyOrder), 1>, Matrix3x3> getPolyCoeffsAs(const Point &p, const Vector &d,
                                                                       const Vector &polyNormal,
                                                                       const Intersection *its, int channel=0) const {
        assert(its);
        const float *coeffs = its->polyCoeffs[channel];
        Matrix3x3 transf = Volpath3D::azimuthSpaceTransformNew(-d, polyNormal);
        Eigen::Matrix<float, PolyUtils::nPolyCoeffs(PolyOrder), 1> shapeCoeffs =
            PolyUtils::rotatePolynomialEigen<PolyOrder>(coeffs, transf.row(0), transf.row(1), transf.row(2));
        return std::make_pair(shapeCoeffs, transf);
    }


protected:
    VaeConfig m_config;

    std::vector<std::vector<ConstraintKdTree>> m_trees;
    size_t m_batchSize;
    int m_polyOrder;
    MediumParameters m_averageMedium;
};

MTS_NAMESPACE_END

#endif /* __VAEHELPER_H__ */
