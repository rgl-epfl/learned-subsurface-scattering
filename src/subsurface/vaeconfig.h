#if !defined(__VAE_CONFIG_H__)
#define __VAE_CONFIG_H__


#include <fstream>


#include <mitsuba/render/scene.h>

#include "json.hpp"

using json = nlohmann::json;

MTS_NAMESPACE_BEGIN


/**
 * Data structure for representing the configuration of a VAE sampler
 */
class VaeConfig {
public:
    VaeConfig() { }
    
    VaeConfig(const std::string &configFilePath, const std::string &configAngularPath, const std::string &outputDir) {
        // Load config from JSON file
        std::ifstream inStream(configFilePath);
        std::string datasetName;
        if (inStream) {
            std::cout << "Loading config...\n";
            json j;
            inStream >> j;
            configName = j["args"]["config"].get<std::string>();
            nLatent = j["config0"]["n_latent"];
            shapeFeaturesName = j["config0"]["shape_features_name"].get<std::string>();
            polyOrder = getPolyOrder(shapeFeaturesName);
            nFeatureCoeffs = nChooseK(3 + polyOrder, polyOrder);
            predictionSpace = j["config0"]["prediction_space"].get<std::string>();
            datasetName = j["config0"]["datasetdir"].get<std::string>();
            useLegendre = shapeFeaturesName.find("legendre") != std::string::npos;

            if (useLegendre) { 
                std::cout << "Using Legendre basis\n";
            }
        } else {
            std::cout << "CONFIG NOT FOUND " << configFilePath << std::endl;
        }

        if (configAngularPath != "") {
            std::ifstream inStream2(configAngularPath);
            if (inStream2) {
                json j;
                inStream2 >> j;
                nAngularLatent = j["config"]["n_latent"];
            } else {
                std::cout << "ANGULAR CONFIG NOT FOUND " << configFilePath << std::endl;
            }
        }
        // After reading the configuration, also read the training data statistics to perform feature normalization  on the fly

        std::string featStatsFilePath = outputDir + "/datasets/" + datasetName + "/train/data_stats.json";
        std::ifstream inStreamStats(featStatsFilePath);
        if (inStreamStats) {
            std::cout << "Loading statistics...\n";

            inStreamStats >> stats;
        } else {
            std::cout << "STATS NOT FOUND " << featStatsFilePath << std::endl;
        }
    }
    bool useLegendre = false;
    int nLatent, nAngularLatent = 2, polyOrder, nFeatureCoeffs;
    std::string shapeFeaturesName, predictionSpace, configName;
    json stats;

private:
    int nChooseK(int n, int k) {
        float result = 1.0f;
        for (int i = 1; i <= k; ++i) {
            result *= (float) (n - (k - i)) / ((float) i);
        }
        return std::round(result);
    }
    
    int getPolyOrder(const std::string &feat_name) {
        return std::stoi(feat_name.substr(feat_name.find_first_of("0123456789")));
    }


};


MTS_NAMESPACE_END

#endif /* __VAE_CONFIG_H__ */
