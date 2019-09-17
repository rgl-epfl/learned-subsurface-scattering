#if !defined(__VAEHELPERPT_H__)
#define __VAEHELPERPT_H__

#include <memory>

#include <mitsuba/core/tls.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sss_particle_tracer.h>

#include "vaeconfig.h"
#include "vaehelper.h"

MTS_NAMESPACE_BEGIN

/**
 * Helper Class to sample using the VAE
 */
class VaeHelperPtracer : public VaeHelper {
public:
    mutable double totalScatterTime = 0.0;
    mutable double numScatterEvaluations = 0.0;

    VaeHelperPtracer(bool usePolynomials, bool useDifftrans, bool disableProjection, Float kernelEpsScale) {
        m_use_polynomials = usePolynomials;
        m_use_difftrans = useDifftrans;
        m_disable_projection = disableProjection;
        m_kernelEpsScale = kernelEpsScale;
    }

    virtual ~VaeHelperPtracer() {
        std::cout << "avgScatterTime: " << totalScatterTime / numScatterEvaluations << std::endl;
        std::cout << "numScatterEvaluations: " << numScatterEvaluations << std::endl;
    }

    virtual bool prepare(const Scene *scene, const std::vector<Shape *> &shapes, const Spectrum &sigmaT,
                         const Spectrum &albedo, float g, float eta, const std::string &modelName,
                         const std::string &absModelName, const std::string &angularModelName,
                         const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig) override;

    virtual ScatterSamplingRecord sample(const Scene *scene, const Point &p, const Vector &d, const Vector &polyNormal, const Spectrum &sigmaT,
                        const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
                        bool projectSamples, int channels) const override;

private:
    bool m_use_difftrans, m_use_polynomials, m_disable_projection;
    Float m_kernelEpsScale;
};

MTS_NAMESPACE_END

#endif /* __VAEHELPERPT_H__ */
