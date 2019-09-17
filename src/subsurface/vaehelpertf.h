#if !defined(__VAEHELPERTF_H__)
#define __VAEHELPERTF_H__

#ifndef USEXLA
#include <tensorflow/core/util/command_line_flags.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#else
#include "model_absorption.h"
#include "model_out_pos_gen.h"
#include "model_out_pos_gen_batched.h"
#endif

#include <mitsuba/render/scene.h>
#include <mitsuba/render/sss_particle_tracer.h>
#include <mitsuba/core/tls.h>

#include "vaeconfig.h"
#include "vaehelper.h"


MTS_NAMESPACE_BEGIN


class SampleTensors;
/**
 * Helper Class to sample using the VAE
 */
class VaeHelperTf : public VaeHelper {
public:

    mutable double totalMsGetPolyCoeffs = 0.0;
    mutable double totalRunScatterNetwork = 0.0;
    mutable double totalRunAbsorptionNetwork = 0.0;
    mutable double totalProjectSamples = 0.0;
    mutable double totalSetupTime = 0.0;
    mutable double numScatterEvaluations = 0.0;


    ~VaeHelperTf() {
        std::cout << "totalMsGetPolyCoeffs: " << totalMsGetPolyCoeffs << std::endl;
        std::cout << "totalRunScatterNetwork: " << totalRunScatterNetwork << std::endl;
        std::cout << "avgRunScatterNetwork: " << totalRunScatterNetwork / numScatterEvaluations << std::endl;
        std::cout << "numScatterEvaluations: " << numScatterEvaluations << std::endl;
        std::cout << "totalRunAbsorptionNetwork: " << totalRunAbsorptionNetwork << std::endl;
        std::cout << "totalProjectSamples: " << totalProjectSamples << std::endl;
        std::cout << "totalSetupTime: " << totalSetupTime << std::endl;
    }

    virtual bool prepare(const Scene *scene, const std::vector<Shape*> &shapes, const Spectrum &sigmaT,
                 const Spectrum &albedo, float g, float eta, const std::string &modelName,
                 const std::string &absModelName, const std::string &angularModelName,
                 const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig) override;

    virtual void sampleBatched(const Scene *scene,
            const Point &p, const Vector &d, const Spectrum &sigmaT, const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its,
            int nSamples, bool projectSamples,
            std::vector<ScatterSamplingRecord> &sRec) const override;
private:

#ifndef USEXLA
    tensorflow::Tensor m_shapeFeatMean, m_shapeFeatStdInv, m_shapeFeatWsMean,
                        m_shapeFeatWsStdInv, m_outPosMean, m_outPosStdInv,
                        m_albedoMean, m_albedoStdInv, m_gMean, m_gStdInv;

    std::shared_ptr<tensorflow::Session> session, absSession, angularSession;
#else
    // mutable ThreadLocal<Model> m_model;
    void setStatistics(Absorption &model) const;
    void setStatistics(Out_pos_gen &model) const;
    void setStatistics(Out_pos_gen_batched &model) const;

#endif



#ifndef USEXLA
    // float estimateAbsorption(const Point &p, const std::vector<float> &shapeCoeffs, const Spectrum &albedo, float g, float eta) const;
#endif
    bool m_hasPhaseVariable = false;
    bool m_absHasPhaseVariable = false;
    bool m_hasAbsModel = false;
    bool m_hasAngularModel = false;
    mutable ThreadLocal<SampleTensors> m_sampleTensors;
};


class SampleTensors : public Object {
public:
    SampleTensors(const VaeConfig &config, size_t batchSize) {
        long long batchSizeLong = batchSize;
        shapeCoeffs = std::vector<float>(VaeHelper::numPolynomialCoefficients(config.polyOrder), 0.0f);
        tmpCoeffs = std::vector<float>(VaeHelper::numPolynomialCoefficients(config.polyOrder), 0.0f);
        tfZlatentBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong, (long long) config.nLatent});
        tfAlbedoBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong, 1});
        tfGBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong, 1});
        tfEtaBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong, 1});
        tfPointsBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong, 3});
        tfInDirsBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong, 3});
        tfShapeCoeffsBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong, config.nFeatureCoeffs});
        tfZlatent1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, config.nLatent});
        tfAlbedo1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, 1});
        tfG1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, 1});
        tfEta1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, 1});
        tfPoints1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, 3});
        tfInDirs1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, 3});
        tfShapeCoeffs1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, config.nFeatureCoeffs});
        tfPhase = tensorflow::Tensor(tensorflow::DataTypeToEnum<bool>::v(), tensorflow::TensorShape());

        tfAngularLatentBatch = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{batchSizeLong,
                                                  (long long) config.nAngularLatent});
        tfAngularLatent1 = tensorflow::Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1, config.nAngularLatent});
    }
    std::vector<float> shapeCoeffs, tmpCoeffs;
    tensorflow::Tensor tfZlatentBatch, tfAngularLatentBatch, tfAngularLatent1, tfAlbedoBatch, tfGBatch, tfEtaBatch, tfPointsBatch, tfInDirsBatch,
                 tfShapeCoeffsBatch, tfZlatent1, tfAlbedo1, tfG1, tfEta1, tfPoints1, tfInDirs1, tfShapeCoeffs1, tfPhase;
};



MTS_NAMESPACE_END

#endif /* __VAEHELPERTF_H__ */
