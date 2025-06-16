#ifndef IMAGE_PREPROCESSING_CONFIGURATION_PARSER_HPP_
#define IMAGE_PREPROCESSING_CONFIGURATION_PARSER_HPP_


#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "configuration/filter/filter.hpp"



namespace configuration {

using ImageFilterPtr = std::unique_ptr<ImageFilter>;

struct FilterPipelineParams {
    std::vector<ImageFilterPtr> filters;
    int numThreads = 1;
    std::string in;
    std::string out;

    void log() {

        std::string filtersInfo;
        for(std::size_t i = 0; i+1 < filters.size(); ++i) {
            filtersInfo += filters[i]->ToString() + ",";
        }
        if (!filters.empty()) {
            filtersInfo += filters[filters.size() - 1]->ToString();
        }

        std::cout << "FilterPipelineParams:\n" 
            << "\tnumThreads=" + std::to_string(numThreads) << "\n" 
            << "\tin=" + in << "\n" 
            << "\tout=" + out << "\n" 
            << "\tfilters=" + filtersInfo << "\n\n"; 
    }

}; 

FilterPipelineParams parse(const std::string& configPath);

FilterPipelineParams parse(nlohmann::json& json);

} // namespace configuration

#endif