#include "vaehelperpt.h"

#include <chrono>

#include <mitsuba/render/mediumparameters.h>
#include <mitsuba/render/polynomials.h>
#include <mitsuba/render/sss_particle_tracer.h>

MTS_NAMESPACE_BEGIN

bool VaeHelperPtracer::prepare(const Scene *scene, const std::vector<Shape *> &shapes, const Spectrum &sigmaT,
                               const Spectrum &albedo, float g, float eta, const std::string &modelName,
                               const std::string &absModelName, const std::string &angularModelName,
                               const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig) {

    m_batchSize     = batchSize;
    m_averageMedium = MediumParameters(albedo, g, eta, sigmaT);
    if (m_use_polynomials) {
        m_polyOrder = pfConfig.order;
        precomputePolynomials(shapes, m_averageMedium, pfConfig);
    }
    return true;
}

ScatterSamplingRecord VaeHelperPtracer::sample(const Scene *scene, const Point &p, const Vector &d, const Vector &polyNormal,
                                               const Spectrum &sigmaT, const Spectrum &albedo, float g, float eta,
                                               Sampler *sampler, const Intersection *its, bool projectSamples,
                                               int channel) const {
    Vector inDir = -d;

    if (m_use_difftrans)
        inDir = -its->shFrame.toWorld(warp::squareToCosineHemisphere(sampler->next2D()));

    Spectrum albedoChannel(albedo[channel]);
    Spectrum sigmaTChannel(sigmaT[channel]);
    MediumParameters medium(albedoChannel, g, eta, sigmaTChannel);
    PolyUtils::Polynomial polynomial;
    bool useLocalDir = false;
    Transform transf;
    float kernelEps;
    if (m_use_polynomials) {
        polynomial.coeffs.resize(its->nPolyCoeffs);
        std::vector<float> tmpCoeffs(its->nPolyCoeffs);
        getPolyCoeffs(its->p, -inDir, sigmaT[channel], g, albedo, its, useLocalDir, polynomial.coeffs, tmpCoeffs, false,
                      channel);
        kernelEps              = PolyUtils::getKernelEps(medium, channel, m_kernelEpsScale);
        polynomial.refPos      = its->p;
        polynomial.refDir      = inDir;
        polynomial.useLocalDir = useLocalDir;
        polynomial.scaleFactor = PolyUtils::getFitScaleFactor(kernelEps);
        polynomial.order       = m_polyOrder;
    }

    Ray ray = Ray(its->p, inDir, ShadowEpsilon, std::numeric_limits<Float>::infinity(), 0.0f);
    if (m_use_polynomials) {
        PolyUtils::adjustRayForPolynomialTracing(ray, polynomial, its->shFrame.n);
    }
    Volpath3D::SamplingConfig config;
    config.disableRR         = false;
    config.ignoreZeroScatter = true;
    config.polyCfg.hardSurfaceConstraint = true;
    config.polyCfg.order = 3;
    ScatterSamplingRecord sRec;
    Volpath3D::PathSampleResult r;
    if (!m_use_polynomials) {
        auto samplePathStart = std::chrono::steady_clock::now();
        r                    = Volpath3D::samplePath(scene, nullptr, sampler, ray, medium, config);
        auto samplePathEnd   = std::chrono::steady_clock::now();
        auto samplePathDiff  = samplePathEnd - samplePathStart;
        totalScatterTime += std::chrono::duration<double, std::milli>(samplePathDiff).count();
        numScatterEvaluations += 1.0;
    } else {
        r = Volpath3D::samplePath(nullptr, &polynomial, sampler, ray, medium, config);
    }

    if (r.status == Volpath3D::PathSampleResult::EStatus::EValid) {
        sRec.n          = r.outNormal;
        sRec.p          = r.pOut;
        sRec.throughput = r.throughput;
        sRec.isValid    = true;
        sRec.outDir     = r.dOut;
    } else {
        sRec.isValid = false;
    }

    if (m_use_polynomials && !m_disable_projection) {
        Eigen::VectorXf polyEigen(its->nPolyCoeffs);
        for (int i = 0; i < its->nPolyCoeffs; ++i) {
            polyEigen[i] = polynomial.coeffs[i];
        }
        PolyUtils::projectPointsToSurface(scene, its->p, inDir, sRec, polyEigen, m_polyOrder, useLocalDir,
                                              polynomial.scaleFactor, kernelEps);
    }
    return sRec;
}

MTS_NAMESPACE_END
