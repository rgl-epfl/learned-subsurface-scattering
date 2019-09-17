#include "vaehelper.h"

#include <chrono>

#include <mitsuba/render/mediumparameters.h>
#include <mitsuba/render/sss_particle_tracer.h>
#include <mitsuba/render/polynomials.h>

#if defined(__OSX__)
#  include <dispatch/dispatch.h>
#endif

MTS_NAMESPACE_BEGIN

void VaeHelper::precomputePolynomialsImpl(const std::vector<Shape*> &shapes,
                                          const MediumParameters &medium,
                                          int channel, const PolyUtils::PolyFitConfig &pfConfig) {
    float kernelEps = PolyUtils::getKernelEps(medium, channel, pfConfig.kernelEpsScale);
    ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("independent")));
    for (size_t shapeIdx = 0; shapeIdx < shapes.size(); ++shapeIdx) {
        int nSamples = std::max(int(shapes[shapeIdx]->getSurfaceArea() * 2.0f / kernelEps), 1024);
        std::vector<Point> sampled_p;
        std::vector<Vector> sampled_n;
        for (int i = 0; i < nSamples; ++i) {
            PositionSamplingRecord pRec(0.0f);
            shapes[shapeIdx]->samplePosition(pRec, sampler->next2D()); //TODO: Handle multiple shapes
            sampled_p.push_back(pRec.p);
            sampled_n.push_back(pRec.n);
        }
        m_trees[channel].push_back(ConstraintKdTree());
        m_trees[channel].back().build(sampled_p, sampled_n);
        TriMesh *trimesh = dynamic_cast<TriMesh*>(shapes[shapeIdx]);
        Log(EInfo, "Precomputing Coeffs on Mesh... (%d vertices, %d constraints)", trimesh->getVertexCount(), sampled_p.size());

        if (!trimesh->hasPolyCoeffs())
            trimesh->createPolyCoeffsArray();
        trimesh->setHasRgb(medium.isRgb());
        PolyStorage *polyCoeffs = trimesh->getPolyCoeffs();

        if (!polyCoeffs)
            Log(EError, "poly coeffs null");

        auto polyFittingStart = std::chrono::steady_clock::now();

#if defined(__OSX__)
        dispatch_apply(trimesh->getVertexCount(),  dispatch_get_global_queue(0, 0), ^(size_t i){
#else
        #pragma parallel for
        for (int i = 0; i < trimesh->getVertexCount(); ++i) {
#endif
            PolyUtils::PolyFitRecord pfRec;
            pfRec.p = trimesh->getVertexPositions()[i];
            pfRec.d = trimesh->getVertexNormals()[i];
            pfRec.n = trimesh->getVertexNormals()[i];
            pfRec.kernelEps = kernelEps;
            pfRec.config = pfConfig;
            pfRec.config.useLightspace = false;
            std::vector<Point> pts;
            std::vector<Vector> dirs;
            PolyUtils::Polynomial result;
            std::tie(result, pts, dirs) = PolyUtils::fitPolynomial(pfRec, &(m_trees[channel].back()));
            for (int j = 0; j < result.coeffs.size(); ++j) {
                polyCoeffs[i].coeffs[channel][j] = result.coeffs[j];
                polyCoeffs[i].kernelEps[channel] = kernelEps;
                polyCoeffs[i].nPolyCoeffs = result.coeffs.size();
            }
        }
#if defined(__OSX__)
        );
#endif
        auto polyFittingEnd = std::chrono::steady_clock::now();
        auto polyFittingDuration = polyFittingEnd - polyFittingStart;
        double totalMs = std::chrono::duration<double, std::milli> (polyFittingDuration).count();
        Log(EInfo, "Done precomputing coeffs. Took %f ms", totalMs);
    }
}


void VaeHelper::precomputePolynomials(const std::vector<Shape*> &shapes, const MediumParameters &medium, const PolyUtils::PolyFitConfig &pfConfig) {
    m_trees.push_back(std::vector<ConstraintKdTree>());
    if (medium.isRgb()) {
        m_trees.push_back(std::vector<ConstraintKdTree>());
        m_trees.push_back(std::vector<ConstraintKdTree>());

        precomputePolynomialsImpl(shapes, medium, 0, pfConfig);
        precomputePolynomialsImpl(shapes, medium, 1, pfConfig);
        precomputePolynomialsImpl(shapes, medium, 2, pfConfig);
    } else {
        precomputePolynomialsImpl(shapes, medium, 0, pfConfig);
    }
    std::cout << "DONE PREPPING\n";
}

bool VaeHelper::prepare(const Scene *scene, const std::vector<Shape *> &shapes, const Spectrum &sigmaT,
                        const Spectrum &albedo, float g, float eta, const std::string &modelName,
                        const std::string &absModelName, const std::string &angularModelName,
                        const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig) {

    m_batchSize = batchSize;

    // Build acceleration data structure to compute polynomial fits efficiently
    MediumParameters medium(albedo, g, eta, sigmaT);
    m_averageMedium      = medium;

    if (modelName == "None") {
        m_polyOrder = pfConfig.order;
        precomputePolynomials(shapes, medium, pfConfig);
        return true; // Dont load any ML models
    }
    std::string modelPath         = outputDir + "models/" + modelName + "/";
    std::string absModelPath      = outputDir + "models_abs/" + absModelName + "/";
    std::string angularModelPath  = outputDir + "models_angular/" + angularModelName + "/";
    std::string configFile        = modelPath + "training-metadata.json";
    std::string angularConfigFile = angularModelPath + "training-metadata.json";

    m_config    = VaeConfig(configFile, angularModelName != "None" ? angularConfigFile : "", outputDir);
    m_polyOrder = m_config.polyOrder;
    precomputePolynomials(shapes, medium, pfConfig);
    return true;
}

std::vector<float> VaeHelper::getPolyCoeffs(const Point &p, const Vector &d,
                                            Float sigmaT_scalar, Float g, const Spectrum &albedo,
                                            const Intersection *its, bool useLightSpace,
                                            std::vector<float> &shapeCoeffs,
                                            std::vector<float> &tmpCoeffs, bool useLegendre, int channel) const {
    if (its) {
        for (auto j = 0; j < its->nPolyCoeffs; ++j) {
            shapeCoeffs[j] = its->polyCoeffs[channel][j];
        }

        // If we predict in light space: rotate the restored poly coefficients
        if (useLightSpace) {
            Vector s, t;
            Vector n = -d;
            Volpath3D::onbDuff(n, s, t);
            switch (m_polyOrder) {
                case 2:
                    PolyUtils::rotatePolynomial<2>(shapeCoeffs, tmpCoeffs, s, t, n);
                    break;
                case 3:
                    PolyUtils::rotatePolynomial<3>(shapeCoeffs, tmpCoeffs, s, t, n);
                    break;
                case 4:
                    PolyUtils::rotatePolynomial<4>(shapeCoeffs, tmpCoeffs, s, t, n);
                    break;
                default:
                    std::cout << "ERROR: Unsupported poly order\n";
            }
        }
    }
    if (useLegendre) {
        std::cout << "CURRENTLY NOT SUPPORTED\n";
        // PolyUtils::legendreTransform(shapeCoeffs);
    }
    return shapeCoeffs;
}


MTS_NAMESPACE_END
