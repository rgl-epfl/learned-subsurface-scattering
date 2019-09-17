#if !defined(__VAEHELPEREIGEN_H__)
#define __VAEHELPEREIGEN_H__

#include <memory>

#include <mitsuba/core/tls.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sss_particle_tracer.h>

#include "vaeconfig.h"
#include "vaehelper.h"
#include "scattereigen.h"

MTS_NAMESPACE_BEGIN

/**
 * Helper Class to sample using the VAE
 */
class VaeHelperEigen : public VaeHelper {
public:

    mutable double totalMsGetPolyCoeffs = 0.0;
    mutable double totalRunScatterNetwork = 0.0;
    mutable double totalRunAbsorptionNetwork = 0.0;
    mutable double totalProjectSamples = 0.0;
    mutable double totalSetupTime = 0.0;
    mutable double numScatterEvaluations = 0.0;
    mutable double totalSampleTime = 0.0;


    VaeHelperEigen(Float kernelEpsScale) {
        m_kernelEpsScale = kernelEpsScale;
    }

    ~VaeHelperEigen() {
        std::cout << "totalMsGetPolyCoeffs: " << totalMsGetPolyCoeffs << std::endl;
        std::cout << "totalRunScatterNetwork: " << totalRunScatterNetwork << std::endl;
        std::cout << "avgRunScatterNetwork: " << totalRunScatterNetwork / numScatterEvaluations << std::endl;
        std::cout << "numScatterEvaluations: " << numScatterEvaluations << std::endl;
        std::cout << "totalRunAbsorptionNetwork: " << totalRunAbsorptionNetwork << std::endl;
        std::cout << "totalProjectSamples: " << totalProjectSamples << std::endl;
        std::cout << "totalSetupTime: " << totalSetupTime << std::endl;
        std::cout << "totalSampleTime: " << totalSampleTime << std::endl;
        std::cout << "AvgSampleTime: " << totalSampleTime / numScatterEvaluations<< std::endl;
    }


    virtual bool prepare(const Scene *scene, const std::vector<Shape *> &shapes, const Spectrum &sigmaT,
                         const Spectrum &albedo, float g, float eta, const std::string &modelName,
                         const std::string &absModelName, const std::string &angularModelName,
                         const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig) override;
    virtual void sampleBatched(const Scene *scene, const Point &p, const Vector &d, const Spectrum &sigmaT,
                        const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
                        int nSamples, bool projectSamples, std::vector<ScatterSamplingRecord> &sRec) const override;
    virtual ScatterSamplingRecord sample(const Scene *scene, const Point &p, const Vector &d,  const Vector &polyNormal, const Spectrum &sigmaT,
                        const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
                        bool projectSamples, int channel=0) const override;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    AbsorptionModel<3> absModel;
    std::unique_ptr<ScatterModelBase> scatterModel;
    Spectrum m_effectiveAlbedo;
    Float m_kernelEpsScale;
};

MTS_NAMESPACE_END

#endif /* __VAEHELPEREIGEN_H__ */
