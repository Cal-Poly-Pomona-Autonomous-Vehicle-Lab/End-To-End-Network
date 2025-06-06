#include "inference.h" 
#include <iostream> 

Inference::Inference(const std::string &config_file, const std::string &weights_file
    const std::string &names_file) {
    
    nn.init(config_file, weights_file, names_file);        

}