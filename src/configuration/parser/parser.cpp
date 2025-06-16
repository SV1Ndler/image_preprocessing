#include "configuration/parser/parser.hpp"

#include <fstream>


namespace configuration {

FilterPipelineParams parse(const std::string& configPath) {
    std::ifstream f(configPath);
    nlohmann::json json = nlohmann::json::parse(f);

    return parse(json);
}

FilterPipelineParams parse(nlohmann::json& json) {
    FilterPipelineParams result;

    result.numThreads = json.value("num_threads", 1);
    result.in = json.value("in", "in.png");
    result.out = json.value("out", "out.png");

    for (const auto& filterConfig : json.at("filters")) {
        const std::string type = filterConfig.at("type").get<std::string>();
            
            if (type == "Sobel") {
                result.filters.push_back(std::make_unique<SobelFilter>());
            }
            else if (type == "Prewitt") {
                result.filters.push_back(std::make_unique<PrewittFilter>());
            }
            else if (type == "Threshold") {
                const int threshold = filterConfig.value("threshold", 128);
                result.filters.push_back(std::make_unique<ThresholdFilter>(threshold));
            }
            else if (type == "Mean") {
                const int kernelSize = filterConfig.value("kernel_size", 3);
                result.filters.push_back(std::make_unique<MeanFilter>(kernelSize));
            }
            else if (type == "Median") {
                const int kernelSize = filterConfig.value("kernel_size", 3);
                result.filters.push_back(std::make_unique<MedianFilter>(kernelSize));
            }
            else {
                throw std::runtime_error("Unknown filter type: " + type);
            }
    }
    
    return result;
}

}