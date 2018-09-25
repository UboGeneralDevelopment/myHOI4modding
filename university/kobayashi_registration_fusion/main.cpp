#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>
#include <fstream>
#include "Eigen/Core"
#include "cpptoml.h"
#include "volume.h"
#include "registration.h"
#include "cuda_runtime.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

//CTボリュームのパス
std::string VGI[2] = { 
    "C:\\Users\\kobayashi\\Documents\\data\\CT\\step-cylinder-al-inBall1 2017-1-12 20-54.vgi", 
    "C:\\Users\\kobayashi\\Documents\\data\\CT\\step-cylinder-al-inBall2 2017-1-16 14-54.vgi"
};
//AffectedRegionでreco_dll.logと同じディレクトリに生成されたファイルのパス
std::string AFF[2] = {
    "C:\\Users\\kobayashi\\Documents\\data\\CT\\step-cylinder-al-inBall1 2017-1-12 20-54\\Affection.toml",
    "C:\\Users\\kobayashi\\Documents\\data\\CT\\step-cylinder-al-inBall2 2017-1-16 14-54\\Affection.toml"
};

template<typename T>
T toml_get(std::shared_ptr<cpptoml::table> table, const std::string &key)
{
    auto k = table->get_qualified_as<T>(key);
    if (k) {
        return *k;
    }
    else {
        throw std::runtime_error("value not found: " + key);
    }
}


int main(int argc, char **argv)
{

  

    Volume vol_a, vol_b;
   
    vol_a.LoadVgi(VGI[0]);
    vol_b.LoadVgi(VGI[1]);
    Volume aff_a, aff_b;

    try {
        auto config_a = cpptoml::parse_file(AFF[0]);
        auto config_b = cpptoml::parse_file(AFF[1]);
        vol_a.start.x = (float)toml_get<double>(config_a, "CT.Volume.StartX");
        vol_a.start.y = (float)toml_get<double>(config_a, "CT.Volume.StartY");
        vol_a.start.z = (float)toml_get<double>(config_a, "CT.Volume.StartZ");
        aff_a.LoadRaw(
            toml_get<std::string>(config_a, "Affection.File"), 0,
            (size_t)toml_get<int64_t>(config_a, "Affection.SizeX"),
            (size_t)toml_get<int64_t>(config_a, "Affection.SizeY"),
            (size_t)toml_get<int64_t>(config_a, "Affection.SizeZ"));
        aff_a.start.x = (float)toml_get<double>(config_a, "Affection.StartX");
        aff_a.start.y = (float)toml_get<double>(config_a, "Affection.StartY");
        aff_a.start.z = (float)toml_get<double>(config_a, "Affection.StartZ");
        
        aff_a.vsize.x = (float)toml_get<double>(config_a, "Affection.Pitch");
        aff_a.vsize.y = (float)toml_get<double>(config_a, "Affection.Pitch");
        aff_a.vsize.z = (float)toml_get<double>(config_a, "Affection.Pitch");

        vol_b.start.x = (float)toml_get<double>(config_b, "CT.Volume.StartX");
        vol_b.start.y = (float)toml_get<double>(config_b, "CT.Volume.StartY");
        vol_b.start.z = (float)toml_get<double>(config_b, "CT.Volume.StartZ");
        aff_b.LoadRaw(
            toml_get<std::string>(config_b, "Affection.File"), 0,
            (size_t)toml_get<int64_t>(config_b, "Affection.SizeX"),
            (size_t)toml_get<int64_t>(config_b, "Affection.SizeY"),
            (size_t)toml_get<int64_t>(config_b, "Affection.SizeZ"));
        aff_b.start.x = (float)toml_get<double>(config_b, "Affection.StartX");
        aff_b.start.y = (float)toml_get<double>(config_b, "Affection.StartY");
        aff_b.start.z = (float)toml_get<double>(config_b, "Affection.StartZ");
        
        aff_b.vsize.x = (float)toml_get<double>(config_b, "Affection.Pitch");
        aff_b.vsize.y = (float)toml_get<double>(config_b, "Affection.Pitch");
        aff_b.vsize.z = (float)toml_get<double>(config_b, "Affection.Pitch");
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
        return 0;
    }
    try {
        Registration(vol_a, aff_a, vol_b, aff_b);
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}