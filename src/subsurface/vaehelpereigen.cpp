#include "vaehelpereigen.h"

#include <chrono>

#include <mitsuba/render/sss_particle_tracer.h>

MTS_NAMESPACE_BEGIN

bool VaeHelperEigen::prepare(const Scene *scene, const std::vector<Shape *> &shapes, const Spectrum &sigmaT,
                             const Spectrum &albedo, float g, float eta, const std::string &modelName,
                             const std::string &absModelName, const std::string &angularModelName,
                             const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig) {
    VaeHelper::prepare(scene, shapes, sigmaT, albedo, g, eta, modelName, absModelName, angularModelName, outputDir,
                       batchSize, pfConfig);
    std::string modelPath          = outputDir + "models/" + modelName + "/";
    std::string absModelPath       = outputDir + "models_abs/" + absModelName + "/";
    std::string angularModelPath   = outputDir + "models_angular/" + angularModelName + "/";
    std::string graph_path         = modelPath + "frozen_model.pb";
    std::string abs_graph_path     = absModelPath + "frozen_model.pb";
    std::string angular_graph_path = angularModelPath + "frozen_model.pb";
    std::string configFile         = modelPath + "training-metadata.json";
    std::string angularConfigFile  = angularModelPath + "training-metadata.json";

    std::string absVariableDir     = absModelPath + "/variables/";
    std::string scatterVariableDir = modelPath + "/variables/";
    std::cout << "Loading model " << modelName << std::endl;
    std::cout << "absVariableDir: " << absVariableDir << std::endl;
    std::cout << "scatterVariableDir: " << scatterVariableDir << std::endl;

    std::cout << "m_config.configName: " << m_config.configName << std::endl;


    if (m_config.configName == "FinalSharedLs/AbsSharedSimComplex") {
        std::cout << "Using VAE Shared Similarity Theory Based\n";
        scatterModel = std::unique_ptr<ScatterModelBase>(new ScatterModelSimShared<3, 4>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3", "LS", true));
    } else if (m_config.configName == "VaeFeaturePreSharedSim2/AbsSharedSimComplex") {
        std::cout << "Using VAE Shared Similarity Theory Based\n";
        scatterModel = std::unique_ptr<ScatterModelBase>(new ScatterModelSimShared<3, 4>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3", "LS", false));
    } else if (m_config.configName == "VaeEfficient" || m_config.configName == "VaeEfficientHardConstraint" || m_config.configName == "VaeEfficientSim") {
        std::cout << "Using VAE Efficient\n";
        scatterModel = std::unique_ptr<ScatterModelBase>(new ScatterModelEfficient<3,4>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3"));
    } else {
        std::cout << "CONFIG NOT RECOGNIZED: Using VAE Shared Similarity Theory Based (EPSILON SCALE)\n";
        std::cout << "m_config.predictionSpace: " << m_config.predictionSpace << std::endl;
        scatterModel = std::unique_ptr<ScatterModelBase>(new ScatterModelSimShared<3, 4>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3", m_config.predictionSpace, true));

        // std::cout << "Using VAE Baseline\n";
        // scatterModel = std::unique_ptr<ScatterModelBase>(new ScatterModel<3, 8>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3"));
    }

    m_effectiveAlbedo = Volpath3D::effectiveAlbedo(albedo);
    Log(EInfo, "Done preprocessing");
    return true;
}

// #define GETTIMINGS

void VaeHelperEigen::sampleBatched(const Scene *scene, const Point &p, const Vector &d, const Spectrum &sigmaT,
                            const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
                            int nSamples, bool projectSamples, std::vector<ScatterSamplingRecord> &sRec) const {
    std::cout << "NOT IMPLEMENTED";
}


ScatterSamplingRecord VaeHelperEigen::sample(const Scene *scene, const Point &p, const Vector &d, const Vector &polyNormal, const Spectrum &sigmaT,
                            const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
                            bool projectSamples, int channel) const {

    // {
    //     const Eigen::Vector3f a(0.f, 0.f, 0.f);
    //     const Eigen::Vector3f b(1.f, 0.f, 0.f);
    //     Eigen::VectorXf latent(m_config.nLatent);
    //     for (int i = 0; i < m_config.nLatent; ++i)
    //         latent[i] = 1.0f;
    //     AbsorptionModel<3>::ShapeVector shapeCoeff1;
    //     for (int i = 0; i < shapeCoeff1.size(); ++i)
    //         shapeCoeff1[i] = 1.0f;
    //     Eigen::Vector3f tmp;
    //     float tmp2;
    //     std::tie(tmp, tmp2) = scatterModel->run(a, b, Spectrum(0.5f), 0.5f, 1.3f, Spectrum(10.0f), shapeCoeff1, latent);
    //     std::cout << "tmp2: " << tmp2 << std::endl;
    //     std::cout << "tmp[0]: " << tmp[0] << std::endl;
    //     std::cout << "tmp[1]: " << tmp[1] << std::endl;
    //     std::cout << "tmp[2]: " << tmp[2] << std::endl;
    // }

    auto sampleStart = std::chrono::steady_clock::now();
#ifdef GETTIMINGS
    auto setupStart = std::chrono::steady_clock::now();
#endif
    AbsorptionModel<3>::ShapeVector shapeCoeffEigen;
    AbsorptionModel<3>::ShapeVector shapeCoeffEigenWs;
    Matrix3x3 asTransform;
    if (m_config.predictionSpace == "AS") {
        const float *coeffs = its->polyCoeffs[channel];
        for (int i = 0; i < shapeCoeffEigenWs.size(); ++i)
            shapeCoeffEigenWs[i] = coeffs[i];
        std::tie(shapeCoeffEigen, asTransform) = getPolyCoeffsAs<3>(p, d, polyNormal, its, channel);
    } else {
        shapeCoeffEigen = getPolyCoeffsEigen<3>(
                             p, d, polyNormal, its,
                             m_config.predictionSpace == "LS", m_config.useLegendre, channel);
    }
#ifdef GETTIMINGS
    auto setupEnd = std::chrono::steady_clock::now();
    auto setupDiff = setupEnd - setupStart;
    totalMsGetPolyCoeffs += std::chrono::duration<double, std::milli> (setupDiff).count();
#endif

    const Eigen::Vector3f inPos(p.x, p.y, p.z);
    const Eigen::Vector3f inDir(d.x, d.y, d.z);

    Spectrum albedoChannel(albedo[channel]);
    Spectrum sigmaTChannel(sigmaT[channel]);
    MediumParameters medium(albedoChannel, g, eta, sigmaTChannel);

    float kernelEps = PolyUtils::getKernelEps(m_averageMedium, channel, m_kernelEpsScale);
    float fitScaleFactor = PolyUtils::getFitScaleFactor(kernelEps);

#ifdef GETTIMINGS
    auto scatteringStart = std::chrono::steady_clock::now();
#endif

    float absorption;
    Eigen::Vector3f outPos;
    std::tie(outPos, absorption) = scatterModel->run(inPos, inDir, medium.albedo, medium.g, medium.eta, medium.sigmaT,
                                                    fitScaleFactor, shapeCoeffEigen, sampler, asTransform);
#ifdef GETTIMINGS
    auto scatteringEnd = std::chrono::steady_clock::now();
    auto scatteringDiff = scatteringEnd - scatteringStart;
    totalRunScatterNetwork += std::chrono::duration<double, std::milli> (scatteringDiff).count();
#endif
    ScatterSamplingRecord sRec;
    sRec.throughput     = Spectrum(1.0f - absorption);
    sRec.outDir         = Vector(1.0f);
    Point sampledP(outPos[0], outPos[1], outPos[2]);
    sRec.p       = sampledP; // Rescale sampled points using sigmaT
    sRec.isValid = absorption < 1.0f;
    numScatterEvaluations++;
#ifdef GETTIMINGS
    auto projectionStart = std::chrono::steady_clock::now();
#endif

    if (m_config.predictionSpace == "AS") {
        PolyUtils::projectPointsToSurface(scene, p, -d, sRec, shapeCoeffEigenWs,
                                                m_config.polyOrder, false, fitScaleFactor, kernelEps);
    } else {
        PolyUtils::projectPointsToSurface(scene, p, -d, sRec, shapeCoeffEigen,
                                                m_config.polyOrder, m_config.predictionSpace == "LS", fitScaleFactor, kernelEps);
    }

#ifdef GETTIMINGS
    auto projectionEnd = std::chrono::steady_clock::now();
    auto projectionDiff = projectionEnd - projectionStart;
    totalProjectSamples += std::chrono::duration<double, std::milli> (projectionDiff).count();
#endif
    auto sampleEnd = std::chrono::steady_clock::now();
    auto sampleDiff = sampleEnd - sampleStart;
    totalSampleTime += std::chrono::duration<double, std::milli> (sampleDiff).count();

    return sRec;
}


MTS_NAMESPACE_END
