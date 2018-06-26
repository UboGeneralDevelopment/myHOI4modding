#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.hpp"
#include "device_launch_parameters.h"

#include "math_functions.h"

#include "vector_operations.hpp"
#include <iostream>


//C++では、includeに～.hはつけない。
#include <iostream>
#include <string>//std::stringを使うのに必要
#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstring>//size_tに使うのに必要
#include <algorithm>
#include <cassert>

#ifndef M_PI
__constant__ const float M_PI = 3.141592;
#endif

__constant__ const size_t ANGLE_DIVISION_THETA = 300;
__constant__ const size_t ANGLE_DIVISION_PHI = 300;


struct Params
{
    size_t voxel_num;
    //float box_size;
    //float voxel_size;

    size_t pixels_width, pixels_height;
    size_t pixels_height_roi;
    float pixel_pitch_x, pixel_pitch_y;
    float source_object_distance;
    float source_detector_distance;
    size_t number_of_projections;
};


__device__ __host__ size_t pixelproc(size_t ix_proj, size_t ix_x, size_t ix_y, const Params &params);

__global__ void pixelproc_kernel(size_t *indicis, size_t ix_proj, Params params)
{
    size_t ix_x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t ix_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix_x < params.pixels_width && ix_y < params.pixels_height) {
        size_t buf_ix = ix_y * params.pixels_width + ix_x;
        indicis[buf_ix] = pixelproc(ix_proj, ix_x, ix_y, params);
    }
}

__device__ __host__ size_t pixelproc(size_t ix_proj, size_t ix_x, size_t ix_y, const Params &params)
{
    float3 xray_source_m = { -params.source_object_distance, 0.0f, 0.0f };
    float3 detector_center_m = {params.source_detector_distance - params.source_object_distance, 0.0f, 0.0f };
    float detector_width = params.pixels_width * params.pixel_pitch_x;
    float detector_height = params.pixels_height * params.pixel_pitch_y;
    float3 detector_topleft_corner_m = detector_center_m + make_float3(0.0f, -detector_width / 2, detector_height / 2);
     
    float rotation = 2 * M_PI * ix_proj / params.number_of_projections;
    float3 ray_hitpoint_m =
        detector_topleft_corner_m + 
        make_float3(0.0f, 0.5f * params.pixel_pitch_x, 0.5f * params.pixel_pitch_y) +
        make_float3(0.0f, ix_x * params.pixel_pitch_x, ix_y * params.pixel_pitch_y);
    float3 raydir_m = ray_hitpoint_m - xray_source_m;
    float3 raydir_obj = rotateAroundZ(raydir_m, -rotation);
    normalize(raydir_obj);
    float sin_theta = raydir_obj.z;
    float phi = atan2f(raydir_obj.y, raydir_obj.x);
    size_t phi_ix = (phi + M_PI) * ANGLE_DIVISION_PHI / (2 * M_PI);
    size_t sin_ix = (sin_theta + 1) * ANGLE_DIVISION_THETA / 2;
    return sin_ix * ANGLE_DIVISION_PHI + phi_ix;
}

int main(int, char **)
{
    Params params;
    /*パラメータここから
    params.pixels_width = 1024;
    params.pixels_height = 1024;
    params.pixels_height_roi = params.pixels_height; //画面下部に回転台が写っている場合に設定
    params.pixel_pitch_x = 0.4f;
    params.pixel_pitch_y = 0.4f;
    params.source_object_distance = 1005.736f; //hairdryer
    params.source_detector_distance = 1519.374f;
    params.number_of_projections = 700; //dryer
    std::string file_prefix = "D:\\TanStorage1\\Data\\test\controller-bin4 2017-3-17 17-47\\Corrected";
    パラメータここまで*/

	/*パラメータここから*/
	params.pixels_width = 512;
	params.pixels_height = 512;
	params.pixels_height_roi = params.pixels_height; //画面下部に回転台が写っている場合に設定
	params.pixel_pitch_x = 0.8f;
	params.pixel_pitch_y = 0.8f;
	params.source_object_distance = 605.732f;
	params.source_detector_distance = 1519.337f;
	params.number_of_projections =500;
	std::string file_prefix = "D:\\TanStorage1\\Data\\!TEST\\controller-bin4 2017-3-17 17-47\\corrected\\Corrected";
	/*パラメータここまで*/

    std::string file_suffix = ".uint16";

    params.voxel_num = 100;
    size_t image_size = params.pixels_width * params.pixels_height;
    uint16_t *buffer = new uint16_t[image_size];//uint16_tはなんだかよくわからんが定義されているらしい。
    size_t *indicis = new size_t[image_size];
    float *altitude_scores = new float[ANGLE_DIVISION_THETA * ANGLE_DIVISION_PHI];//ここで、角度ごとの透過率を格納するaltitude_scoresを作る

    for (size_t i = 0; i < ANGLE_DIVISION_THETA * ANGLE_DIVISION_PHI; i++) {
        altitude_scores[i] = 0.0f;
    }//altitude_scoresの初期化

    size_t *indicis_dev;
    cudaMalloc(&indicis_dev, sizeof(size_t)* image_size);
    std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;//ここ不明

    uint16_t *buffer_dev;
    cudaMalloc(&buffer_dev, sizeof(uint16_t) * image_size);
    std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;//ここも不明


    float3 v = make_float3(0, 0, 0);
    for (size_t ix_proj = 0; ix_proj < 1 * params.number_of_projections; ix_proj++) {

        //read
        {
            std::string filename = file_prefix + std::to_string(ix_proj) + file_suffix;
            std::cout << "loading " << ix_proj << std::endl;
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs.is_open()) {
                std::cerr << "file not found : " << filename << std::endl;
                continue;
            }
            ifs.read((char *)buffer, sizeof(uint16_t) * image_size);
            if (ifs.fail()) {
                std::cerr << "file too short : " << filename << std::endl;
                continue;
            }
        }

        cudaMemcpy(buffer_dev, buffer, sizeof(uint16_t) * image_size, cudaMemcpyHostToDevice);
        //std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;

        dim3 threadsPerBlock(16, 16);
        dim3 blocks(params.pixels_width / threadsPerBlock.x + 1, params.pixels_height / threadsPerBlock.y + 1);
        pixelproc_kernel<<<blocks,threadsPerBlock>>>(indicis_dev, ix_proj, params);
        cudaMemcpy(indicis, indicis_dev, sizeof(size_t)* image_size, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < image_size; i++) {
            size_t y = i / params.pixels_width;
            if (y >= params.pixels_height_roi) {
                continue;
            }
            float rate = buffer[i] / 65536.0f;
            float len = -log(rate);
            size_t ix = indicis[i];
            altitude_scores[ix] += len * len;
        }

        
        std::cout << ix_proj << std::endl;
    }

 
    int maxi = 0;
    for (int i = 0; i < ANGLE_DIVISION_PHI * ANGLE_DIVISION_THETA; i++) {
        if (altitude_scores[i] > altitude_scores[maxi]) {
            maxi = i;
        }
    }
    const float maxv = altitude_scores[maxi];


    //size_t phi_ix = (phi + M_PI) * ANGLE_DIVISION_PHI / (2 * M_PI);
    //size_t sin_ix = (sin_theta + 1) * ANGLE_DIVISION_THETA / 2;
    int phi_ix = maxi % ANGLE_DIVISION_PHI;
    int sin_ix = maxi / ANGLE_DIVISION_PHI;
    const float worstdir_phi = -M_PI + 2.0f * M_PI *phi_ix / ANGLE_DIVISION_PHI;
    const float worstdir_sintheta = -1 + 2.0f * sin_ix / ANGLE_DIVISION_THETA;
    const float worstdir_theta = std::asin(worstdir_sintheta);
    std::cout << sin_ix << "/" << ANGLE_DIVISION_THETA << ' ' << phi_ix << '/' << ANGLE_DIVISION_PHI << std::endl;
    std::cout << "theta " << worstdir_theta * 180 / M_PI << "[deg] phi " << worstdir_phi  * 180 / M_PI << " [deg]" << std::endl;
    std::cout << maxi << ' ' << maxv << std::endl;

   
    size_t ix_score = 0;
    for (int ix_t = 0; ix_t < ANGLE_DIVISION_THETA; ix_t++) {
        for (int ix_p = 0; ix_p < ANGLE_DIVISION_PHI; ix_p++, ix_score++) {
            float score = altitude_scores[ix_score];
            if (score > 0) {
                float z = 2.0f * ix_t / ANGLE_DIVISION_THETA - 1.0;
                float phi = 2.0f * M_PI * ix_p / ANGLE_DIVISION_PHI - M_PI;
                float xy_norm = std::sqrt(1 - z * z);
                float x = xy_norm * std::cos(phi);
                float y = xy_norm * std::sin(phi);
            }
        }
    }
 
    delete[] altitude_scores;
    delete[] buffer;
    return 0;
}

