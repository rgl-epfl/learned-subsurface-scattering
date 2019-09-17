#include "vaehelpertf.h"

#include <chrono>

#include <mitsuba/render/sss_particle_tracer.h>

MTS_NAMESPACE_BEGIN

#ifndef USEXLA
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status loadTfGraph(const string& graph_file_name,
            std::shared_ptr<tensorflow::Session>* session, bool &hasPhaseVariable) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    // config = tf.ConfigProto(
    //     device_count = {'GPU': 0}
    // )

    tensorflow::SessionOptions options;

    tensorflow::ConfigProto* config = &options.config;
    // disabled GPU entirely
    (*config->mutable_device_count())["GPU"] = 0;
    // place nodes somewhere
    config->set_allow_soft_placement(true);
    // Turn on JIT, doesnt work?
    // config->mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);
    session->reset(tensorflow::NewSession(options));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }

    hasPhaseVariable = false;
    int node_count = graph_def.node_size();
    for (int i = 0; i < node_count; i++) {
        if (graph_def.node(i).name() == std::string("phase")) {
            hasPhaseVariable = true;
        }
    }
    return Status::OK();
}

#else

void VaeHelperTf::setStatistics(Out_pos_gen &model) const {
    std::vector<float> albedoMean(1), albedoStdInv(1), outPosMean(3), outPosStdInv(3), gMean(1), gStdInv(1),
        shapeFeatMean(m_config.nFeatureCoeffs), shapeFeatStdInv(m_config.nFeatureCoeffs);

    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_mean"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_mean"].size(),
                shapeFeatMean.data());
    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_stdinv"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_stdinv"].size(),
                shapeFeatStdInv.data());
    std::copy_n(m_config.stats["outPosRel" + m_config.predictionSpace + "_mean"].begin(),
                m_config.stats["outPosRel" + m_config.predictionSpace + "_mean"].size(),
                outPosMean.data());
    std::copy_n(m_config.stats["outPosRel" + m_config.predictionSpace + "_stdinv"].begin(),
                m_config.stats["outPosRel" + m_config.predictionSpace + "_stdinv"].size(),
                outPosStdInv.data());
    std::copy_n(m_config.stats["effAlbedo_mean"].begin(),
                m_config.stats["effAlbedo_mean"].size(),
                albedoMean.data());
    std::copy_n(m_config.stats["effAlbedo_stdinv"].begin(),
                m_config.stats["effAlbedo_stdinv"].size(),
                albedoStdInv.data());
    std::copy_n(m_config.stats["g_mean"].begin(),
                m_config.stats["g_mean"].size(),
                gMean.data());
    std::copy_n(m_config.stats["g_stdinv"].begin(),
                m_config.stats["g_stdinv"].size(),
                gStdInv.data());
    model.feedStatistics(
        albedoMean.data(),
        albedoStdInv.data(),
        outPosMean.data(),
        outPosStdInv.data(),
        gMean.data(),
        gStdInv.data(),
        shapeFeatMean.data(),
        shapeFeatStdInv.data());
}

void VaeHelperTf::setStatistics(Out_pos_gen_batched &model) const {
    std::vector<float> albedoMean(1), albedoStdInv(1), outPosMean(3), outPosStdInv(3), gMean(1), gStdInv(1),
        shapeFeatMean(m_config.nFeatureCoeffs), shapeFeatStdInv(m_config.nFeatureCoeffs);

    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_mean"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_mean"].size(),
                shapeFeatMean.data());
    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_stdinv"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_stdinv"].size(),
                shapeFeatStdInv.data());
    std::copy_n(m_config.stats["outPosRel" + m_config.predictionSpace + "_mean"].begin(),
                m_config.stats["outPosRel" + m_config.predictionSpace + "_mean"].size(),
                outPosMean.data());
    std::copy_n(m_config.stats["outPosRel" + m_config.predictionSpace + "_stdinv"].begin(),
                m_config.stats["outPosRel" + m_config.predictionSpace + "_stdinv"].size(),
                outPosStdInv.data());
    std::copy_n(m_config.stats["effAlbedo_mean"].begin(),
                m_config.stats["effAlbedo_mean"].size(),
                albedoMean.data());
    std::copy_n(m_config.stats["effAlbedo_stdinv"].begin(),
                m_config.stats["effAlbedo_stdinv"].size(),
                albedoStdInv.data());
    std::copy_n(m_config.stats["g_mean"].begin(),
                m_config.stats["g_mean"].size(),
                gMean.data());
    std::copy_n(m_config.stats["g_stdinv"].begin(),
                m_config.stats["g_stdinv"].size(),
                gStdInv.data());
    model.feedStatistics(
        albedoMean.data(),
        albedoStdInv.data(),
        outPosMean.data(),
        outPosStdInv.data(),
        gMean.data(),
        gStdInv.data(),
        shapeFeatMean.data(),
        shapeFeatStdInv.data());
}

void VaeHelperTf::setStatistics(Absorption &model) const {
    std::vector<float> albedoMean(1), albedoStdInv(1), gMean(1), gStdInv(1),
        shapeFeatMean(m_config.nFeatureCoeffs), shapeFeatStdInv(m_config.nFeatureCoeffs);

    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_mean"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_mean"].size(),
                shapeFeatMean.data());
    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_stdinv"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_stdinv"].size(),
                shapeFeatStdInv.data());
    std::copy_n(m_config.stats["effAlbedo_mean"].begin(),
                m_config.stats["effAlbedo_mean"].size(),
                albedoMean.data());
    std::copy_n(m_config.stats["effAlbedo_stdinv"].begin(),
                m_config.stats["effAlbedo_stdinv"].size(),
                albedoStdInv.data());
    std::copy_n(m_config.stats["g_mean"].begin(),
                m_config.stats["g_mean"].size(),
                gMean.data());
    std::copy_n(m_config.stats["g_stdinv"].begin(),
                m_config.stats["g_stdinv"].size(),
                gStdInv.data());
    model.feedStatistics(
        albedoMean.data(),
        albedoStdInv.data(),
        gMean.data(),
        gStdInv.data(),
        shapeFeatMean.data(),
        shapeFeatStdInv.data());
}



#endif


bool VaeHelperTf::prepare(const Scene *scene, const std::vector<Shape*> &shapes, const Spectrum &sigmaT,
const Spectrum &albedo, float g, float eta,
const std::string &modelName, const std::string &absModelName, const std::string &angularModelName,
const std::string &outputDir, int batchSize, const PolyUtils::PolyFitConfig &pfConfig) {
    VaeHelper::prepare(scene, shapes, sigmaT, albedo, g, eta, modelName, absModelName, angularModelName, outputDir, batchSize, pfConfig);
    std::string modelPath = outputDir + "models/" + modelName + "/";
    std::string absModelPath = outputDir + "models_abs/" + absModelName + "/";
    std::string angularModelPath = outputDir + "models_angular/" + angularModelName + "/";
    std::string graph_path = modelPath + "frozen_model.pb";
    std::string abs_graph_path = absModelPath + "frozen_model.pb";
    std::string angular_graph_path = angularModelPath + "frozen_model.pb";
    std::string configFile = modelPath + "training-metadata.json";
    std::string angularConfigFile = angularModelPath + "training-metadata.json";

    std::cout << "Loading model " << modelName << std::endl;


    // TENSORFLOW Setup
    // We need to call this to set up global state for TensorFlow.
    // This seems unneeded for simple CPU execution?
    // tensorflow::port::InitMain(argv[0], &argc, &argv);
    // if (argc > 1) {
    //     LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    //     return -1;
    // }
#ifndef USEXLA
    // First we load and initialize the model.
    Status loadGraphStatus = loadTfGraph(graph_path, &session, m_hasPhaseVariable);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << loadGraphStatus;
        return -1;
    }

    if (absModelName != "None") {
        m_hasAbsModel = true;
        loadGraphStatus = loadTfGraph(abs_graph_path, &absSession, m_absHasPhaseVariable);
        if (!loadGraphStatus.ok()) {
            LOG(ERROR) << loadGraphStatus;
            return -1;
        }
    }

    if (angularModelName != "None") {
        m_hasAngularModel = true;
        bool hasPhase;
        loadGraphStatus = loadTfGraph(angular_graph_path, &angularSession, hasPhase);
        if (!loadGraphStatus.ok()) {
            LOG(ERROR) << loadGraphStatus;
            return -1;
        }
    }

    // Collect shapeFeature, outPos and Albedo stats into tensors
    m_shapeFeatMean = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{m_config.nFeatureCoeffs});
    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_mean"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_mean"].size(),
                m_shapeFeatMean.flat<float>().data());
    m_shapeFeatStdInv = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{m_config.nFeatureCoeffs});
    std::copy_n(m_config.stats[m_config.shapeFeaturesName + "_stdinv"].begin(),
                m_config.stats[m_config.shapeFeaturesName + "_stdinv"].size(),
                m_shapeFeatStdInv.flat<float>().data());
    m_shapeFeatWsMean = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{m_config.nFeatureCoeffs});
    std::string degStr = std::to_string(m_config.polyOrder);
    std::copy_n(m_config.stats["mlsPoly" + degStr + "_mean"].begin(),
                m_config.stats["mlsPoly" + degStr + "_mean"].size(),
                m_shapeFeatWsMean.flat<float>().data());
    m_shapeFeatWsStdInv = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{m_config.nFeatureCoeffs});
    std::copy_n(m_config.stats["mlsPoly" + degStr + "_stdinv"].begin(),
                m_config.stats["mlsPoly" + degStr + "_stdinv"].size(),
                m_shapeFeatWsStdInv.flat<float>().data());
    m_outPosMean = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{3});
    std::copy_n(m_config.stats["outPosRel" + m_config.predictionSpace + "_mean"].begin(),
                m_config.stats["outPosRel" + m_config.predictionSpace + "_mean"].size(),
                m_outPosMean.flat<float>().data());
    m_outPosStdInv = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{3});
    std::copy_n(m_config.stats["outPosRel" + m_config.predictionSpace + "_stdinv"].begin(),
                m_config.stats["outPosRel" + m_config.predictionSpace + "_stdinv"].size(),
                m_outPosStdInv.flat<float>().data());
    m_albedoMean = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{3});
    std::copy_n(m_config.stats["effAlbedo_mean"].begin(),
                m_config.stats["effAlbedo_mean"].size(),
                m_albedoMean.flat<float>().data());
    m_albedoStdInv = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{3});
    std::copy_n(m_config.stats["effAlbedo_stdinv"].begin(),
                m_config.stats["effAlbedo_stdinv"].size(),
                m_albedoStdInv.flat<float>().data());
    m_gMean = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1});
    std::copy_n(m_config.stats["g_mean"].begin(),
                m_config.stats["g_mean"].size(),
                m_gMean.flat<float>().data());
    m_gStdInv = Tensor(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{1});
    std::copy_n(m_config.stats["g_stdinv"].begin(),
                m_config.stats["g_stdinv"].size(),
                m_gStdInv.flat<float>().data());
#endif

    Log(EInfo, "Done preprocessing");
    return true;
}


void VaeHelperTf::sample(
    const Scene *scene, const Point &p, const Vector &d, const Vector &polyNormal, const Spectrum &sigmaT,
    const Spectrum &albedo, float g, float eta, Sampler *sampler, const Intersection *its, int nSamples, bool projectSamples,
    std::vector<ScatterSamplingRecord> &sRec) const {

    bool print_timings = false;
    if (nSamples != m_batchSize && nSamples != 1) {
        std::cout << "Cannot generate #n samples: " << nSamples << " batch Size: " << m_batchSize << std::endl;
        return;
    }
    // Extract polynomial for the current sample location and direction
    Float sigmaT_scalar = (sigmaT[0] + sigmaT[1] + sigmaT[2]) / 3.0f;
    struct timespec polyspec;
    clock_gettime(CLOCK_REALTIME, &polyspec);

    if (!m_sampleTensors.get()) {
        m_sampleTensors.set(new SampleTensors(m_config, m_batchSize));
    }



    std::vector<float> &shapeCoeffs = m_sampleTensors.get()->shapeCoeffs;
    std::vector<float> &tmpCoeffs = m_sampleTensors.get()->tmpCoeffs;
    auto getPolyCoeffsStart = std::chrono::steady_clock::now();
    getPolyCoeffs(p, d, sigmaT_scalar, g, albedo, its, m_config.predictionSpace == "LS", shapeCoeffs, tmpCoeffs, m_config.useLegendre);
    auto getPolyCoeffsEnd = std::chrono::steady_clock::now();
    auto getPolyCoeffsDiff = getPolyCoeffsEnd - getPolyCoeffsStart;
    totalMsGetPolyCoeffs += std::chrono::duration<double, std::milli> (getPolyCoeffsDiff).count();

    auto setupStart = std::chrono::steady_clock::now();
    struct timespec polyspec2;
    clock_gettime(CLOCK_REALTIME, &polyspec2);
    if (print_timings)
        std::cout << ((double)(polyspec2.tv_nsec - polyspec.tv_nsec)) << " ns poly" <<  std::endl;

#ifndef USEXLA

    Tensor& tfZlatentBatch = m_sampleTensors.get()->tfZlatentBatch;
    Tensor& tfAngularLatentBatch = m_sampleTensors.get()->tfAngularLatentBatch;
    Tensor& tfAlbedoBatch = m_sampleTensors.get()->tfAlbedoBatch;
    Tensor& tfGBatch = m_sampleTensors.get()->tfGBatch;
    Tensor& tfEtaBatch = m_sampleTensors.get()->tfEtaBatch;
    Tensor& tfPointsBatch = m_sampleTensors.get()->tfPointsBatch;
    Tensor& tfInDirsBatch = m_sampleTensors.get()->tfInDirsBatch;
    Tensor& tfShapeCoeffsBatch = m_sampleTensors.get()->tfShapeCoeffsBatch;

    Tensor& tfZlatent1 = m_sampleTensors.get()->tfZlatent1;
    Tensor& tfAngularLatent1 = m_sampleTensors.get()->tfAngularLatent1;
    Tensor& tfAlbedo1 = m_sampleTensors.get()->tfAlbedo1;
    Tensor& tfG1 = m_sampleTensors.get()->tfG1;
    Tensor& tfEta1 = m_sampleTensors.get()->tfEta1;
    Tensor& tfPoints1 = m_sampleTensors.get()->tfPoints1;
    Tensor& tfInDirs1 = m_sampleTensors.get()->tfInDirs1;
    Tensor& tfShapeCoeffs1 = m_sampleTensors.get()->tfShapeCoeffs1;
    Tensor& tfPhase = m_sampleTensors.get()->tfPhase;

    Tensor& tfZlatent = nSamples == m_batchSize ? tfZlatentBatch : tfZlatent1;
    Tensor& tfAngularLatent = nSamples == m_batchSize ? tfAngularLatentBatch : tfAngularLatent1;
    Tensor& tfAlbedo = nSamples == m_batchSize ? tfAlbedoBatch : tfAlbedo1;
    Tensor& tfG = nSamples == m_batchSize ? tfGBatch : tfG1;
    Tensor& tfEta = nSamples == m_batchSize ? tfEtaBatch : tfEta1;
    Tensor& tfPoints = nSamples == m_batchSize ? tfPointsBatch : tfPoints1;
    Tensor& tfInDirs = nSamples == m_batchSize ? tfInDirsBatch : tfInDirs1;
    Tensor& tfShapeCoeffs = nSamples == m_batchSize ? tfShapeCoeffsBatch : tfShapeCoeffs1;

    tfPhase.scalar<bool>()() = false;

    // Sample latent vars
    sampleGaussianVector(tfZlatent.flat<float>().data(), sampler, nSamples * m_config.nLatent);
    sampleUniformVector(tfAngularLatent.flat<float>().data(), sampler, nSamples * m_config.nAngularLatent);

    // Copy data to feed dict
    float *rawAlbedo = tfAlbedo.flat<float>().data();
    float *rawG = tfG.flat<float>().data();
    float *rawEta = tfEta.flat<float>().data();
    float *rawPoints = tfPoints.flat<float>().data();
    float *rawInDirs = tfInDirs.flat<float>().data();
    float *rawShapeCoeffs = tfShapeCoeffs.flat<float>().data();
    float effectiveAlbedo =  -std::log(1.0f - albedo[0] * (1.0f - std::exp(-8.0f))) / 8.0f;

    for (int i = 0; i < nSamples; ++i) {
        rawAlbedo[i] = effectiveAlbedo;
        rawG[i] = g;
        rawEta[i] = eta;
        rawPoints[3 * i + 0] = p.x;
        rawPoints[3 * i + 1] = p.y;
        rawPoints[3 * i + 2] = p.z;
        rawInDirs[3 * i + 0] = d.x;
        rawInDirs[3 * i + 1] = d.y;
        rawInDirs[3 * i + 2] = d.z;
        std::copy_n(shapeCoeffs.begin(), shapeCoeffs.size(), rawShapeCoeffs + i * shapeCoeffs.size());
    }

    std::vector<std::pair<string, Tensor>> feedDict = {{tensorflow::string("scatterLatent"), tfZlatent}, // (m_batchSize, m_config.nLatent)
                                    //{tensorflow::string("angularLatent"), tfAngularLatent},
                                    {tensorflow::string("shapeFeatures"), tfShapeCoeffs}, // (m_batchSize, nFeatures)
                                    {tensorflow::string("shapeFeaturesMean"), m_shapeFeatMean}, // (nFeatures)
                                    {tensorflow::string("shapeFeaturesStdInv"), m_shapeFeatStdInv}, // (nFeatures)
                                    {tensorflow::string("outPosMean"), m_outPosMean},  // (m_batchSize, 3)
                                    {tensorflow::string("outPosStdInv"), m_outPosStdInv}, // (3)
                                    {tensorflow::string("inPos"), tfPoints}, // (m_batchSize, 3)
                                    {tensorflow::string("effAlbedo"), tfAlbedo}, // (m_batchSize, 1)
                                    {tensorflow::string("effAlbedoMean"), m_albedoMean}, // (1)
                                    {tensorflow::string("effAlbedoStdInv"), m_albedoStdInv},
                                    {tensorflow::string("g"), tfG}, // (m_batchSize, 3)
                                    {tensorflow::string("gMean"), m_gMean}, // (3)
                                    {tensorflow::string("gStdInv"), m_gStdInv},
                                    {tensorflow::string("ior"), tfEta}}; // (m_batchSize, 3)
    if (m_config.predictionSpace == "LS")
        feedDict.push_back({tensorflow::string("inDir"), tfInDirs});

    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);

    Status run_status, run_status_abs;
    if (m_hasPhaseVariable)
        feedDict.push_back({tensorflow::string("phase"), tfPhase});


    auto setupEnd = std::chrono::steady_clock::now();
    auto setupDiff = setupEnd - setupStart;
    totalSetupTime += std::chrono::duration<double, std::milli> (setupDiff).count();

    auto scatterSampleStart = std::chrono::steady_clock::now();
    std::vector<Tensor> outputs;
    run_status = session->Run(feedDict, {"scatter/out_pos_gen"}, {}, &outputs);

    auto scatterSampleEnd = std::chrono::steady_clock::now();
    auto scatterSampleDiff = scatterSampleEnd - scatterSampleStart;
    totalRunScatterNetwork += std::chrono::duration<double, std::milli> (scatterSampleDiff).count();
    numScatterEvaluations += nSamples;

    setupStart = std::chrono::steady_clock::now();

    // either add or remove the phase variable as needed
    if (m_absHasPhaseVariable && !m_hasPhaseVariable)
        feedDict.push_back({tensorflow::string("phase"), tfPhase});
    else if (!m_absHasPhaseVariable && m_hasPhaseVariable)
        feedDict.pop_back();

    feedDict = {{tensorflow::string("shapeFeatures"), tfShapeCoeffs}, // (m_batchSize, nFeatures)
                {tensorflow::string("shapeFeaturesMean"), m_shapeFeatMean}, // (nFeatures)
                {tensorflow::string("shapeFeaturesStdInv"), m_shapeFeatStdInv}, // (nFeatures)
                {tensorflow::string("effAlbedo"), tfAlbedo}, // (m_batchSize, 1)
                {tensorflow::string("effAlbedoMean"), m_albedoMean}, // (1)
                {tensorflow::string("effAlbedoStdInv"), m_albedoStdInv},
                {tensorflow::string("g"), tfG}, // (m_batchSize, 3)
                {tensorflow::string("gMean"), m_gMean}, // (3)
                {tensorflow::string("gStdInv"), m_gStdInv},
                {tensorflow::string("ior"), tfEta}}; // (m_batchSize, 3)

    setupEnd = std::chrono::steady_clock::now();
    setupDiff = setupEnd - setupStart;
    totalSetupTime += std::chrono::duration<double, std::milli> (setupDiff).count();


    auto absorptionStart = std::chrono::steady_clock::now();

    std::vector<Tensor> absOutputs;
    if (m_hasAbsModel) {
        run_status_abs = absSession->Run(feedDict, {"absorption/absorption"}, {}, &absOutputs);
    }

    auto absorptionEnd = std::chrono::steady_clock::now();
    auto absorptionDiff = absorptionEnd - absorptionStart;
    totalRunAbsorptionNetwork += std::chrono::duration<double, std::milli> (absorptionDiff).count();

    std::vector<Tensor> angularOutputs;
    if (m_hasAngularModel) {
        run_status_abs = angularSession->Run(feedDict, {"angular/out_dir_gen"}, {}, &angularOutputs);
    }


    struct timespec spec2;
    clock_gettime(CLOCK_REALTIME, &spec2);
    if (print_timings)
        std::cout << ((double)(spec2.tv_nsec - spec.tv_nsec)) << " ns point" <<  std::endl;

    if (!run_status.ok()) {
        std::cout << "ERROR " << run_status.ToString() << std::endl;
        Log(EError, run_status.ToString().c_str());
    }
    if (m_hasAbsModel && !run_status_abs.ok()) {
        std::cout << "ERROR " << run_status_abs.ToString() << std::endl;
        Log(EError, run_status_abs.ToString().c_str());
    }

    for (int i = 0; i < nSamples; ++i) {
        if (m_hasAbsModel)
            sRec[i].throughput = absOutputs[0].flat<float>().data()[i];
        else
            sRec[i].throughput = 0.0f;

        if (m_hasAngularModel) {
            Vector sampledD(angularOutputs[0].flat<float>().data()[3 * i + 0], angularOutputs[0].flat<float>().data()[3 * i + 1], angularOutputs[0].flat<float>().data()[3 * i + 2]);
            sRec[i].outDir = sampledD;
        } else
            sRec[i].outDir = Vector(1.0f);

        Point sampledP(outputs[0].flat<float>().data()[3 * i + 0], outputs[0].flat<float>().data()[3 * i + 1], outputs[0].flat<float>().data()[3 * i + 2]);
        sRec[i].p = p + (sampledP - p) / sigmaT.average();  // Rescale sampled points using sigmaT
        sRec[i].isValid = true;
    }
#else
    //TODO: Use ThreadLocal storage here
    Absorption absorptionModel;
    Out_pos_gen outPosGenModel;
    Out_pos_gen_batched outPosGenBatchedModel;

    if (!absorptionModel.hasStatistics()) {
        std::cout << "Setting stats abs\n";
        setStatistics(absorptionModel);
    }
    if (!outPosGenModel.hasStatistics()) {
        std::cout << "Setting stats outposgeb\n";
        setStatistics(outPosGenModel);
    }
    if (!outPosGenBatchedModel.hasStatistics()) {
        std::cout << "Setting stats outposgenbatched\n";
        setStatistics(outPosGenBatchedModel);
    }

    float inPosArr[3] = {p.x, p.y, p.z};
    float inDirArr[3] = {d.x, d.y, d.z};
    float effectiveAlbedo =  -std::log(1.0f - albedo[0] * (1.0f - std::exp(-8.0f))) / 8.0f;

    //TODO: Use ThreadLocal storage here
    std::vector<float> latentZ(outPosGenBatchedModel.getBatchSize() * m_config.nLatent);
    std::vector<float> outPos(outPosGenBatchedModel.getBatchSize() * 3);

    float outAbs;

    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);

    sampleGaussianVector(latentZ.data(), sampler, nSamples * m_config.nLatent);

    // TODO: this always computes 8 samples currently

    if (nSamples > 1) {
        outPosGenBatchedModel.run(inPosArr, inDirArr, effectiveAlbedo,
                g, eta, latentZ.data(),
                shapeCoeffs.data(),
                outPos.data());
    } else {
        outPosGenModel.run(inPosArr, inDirArr, effectiveAlbedo,
        g, eta, latentZ.data(),
        shapeCoeffs.data(),
        outPos.data());
    }

    absorptionModel.run(effectiveAlbedo, g, eta, shapeCoeffs.data(), &outAbs);
    struct timespec spec2;
    clock_gettime(CLOCK_REALTIME, &spec2);
    //std::cout << ((double)(spec2.tv_nsec - spec.tv_nsec)) << " ns point " << nSamples << std::endl;
    for (int i = 0; i < nSamples; ++i) {
        sRec[i].throughput = outAbs;

        Point sampledP(outPos[3 * i + 0], outPos[3 * i + 1], outPos[3 * i + 2]);
        sRec[i].p = p + (sampledP - p) / sigmaT[0];  // Rescale sampled points using sigmaT
        sRec[i].isValid = true;
    }
#endif

    auto projectionStart = std::chrono::steady_clock::now();
    if (projectSamples) {
        Eigen::VectorXf polyEigen(shapeCoeffs.size());
        for (int i = 0; i < shapeCoeffs.size(); ++i) {
            polyEigen[i] = shapeCoeffs[i];
        }
        float kernelEps = PolyUtils::getKernelEps(m_medium);
        float fitScaleFactor = PolyUtils::getFitScaleFactor(kernelEps);
        for (int i = 0; i < nSamples; ++i)
            PolyUtils::projectPointsToSurface(scene, p, -d, sRec, polyEigen,
                                                m_config.polyOrder, m_config.predictionSpace == "LS", fitScaleFactor, kernelEps);
    }
    auto projectionEnd = std::chrono::steady_clock::now();
    auto projectionDiff = projectionEnd - projectionStart;
    totalProjectSamples += std::chrono::duration<double, std::milli> (projectionDiff).count();

}


MTS_NAMESPACE_END
