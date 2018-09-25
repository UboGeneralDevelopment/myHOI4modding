
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernel.h"
#include "volume.cuh"



__global__ void affineinv_kernel(
    const uint16_t *input, Shape input_shape, uint16_t *output, Shape output_shape,
    Mat3 rot_inv, Vec3 trans, uint16_t defvar)
{
    int vix_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vix_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vix_z = blockIdx.z * blockDim.z + threadIdx.z;
    Index ix(vix_x, vix_y, vix_z);
    if (!output_shape.InRange(ix)) {
        return;
    }

    Vec3 pos = output_shape.AbsoluteCenter(ix);
    Vec3 pos_original = rot_inv * (pos - trans);
    Index ix_original = input_shape.VoxelIndex(pos_original);
    if (input_shape.InRange(ix_original)) {
        output[output_shape.at(ix)] = input[input_shape.at(ix_original)];
    }
    else {
        output[output_shape.at(ix)] = defvar;
    }
}


__global__ void affineinv_kernel(
    const uint16_t *input, Shape input_shape, uint16_t *output, Shape output_shape,
    Mat3 rot_inv, Vec3 trans, uint16_t defvar, size_t super)
{
    int vix_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vix_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vix_z = blockIdx.z * blockDim.z + threadIdx.z;
    Index ix(vix_x, vix_y, vix_z);
    if (!output_shape.InRange(ix)) {
        return;
    }

   
    Vec3 start_pos = output_shape.Absolute(ix) + output_shape.vsize / super / 2.0f;
    uint32_t sum = 0;
    for (int i = 0; i < super; i++) {
        float x = start_pos.x + i * output_shape.vsize.x / super;
        for (int j = 0; j < super; j++) {
            float y = start_pos.y + j * output_shape.vsize.y / super;
            for (int k = 0; k < super; k++) {
                float z = start_pos.z + k * output_shape.vsize.z / super;
                //Vec3 pos = output_shape.AbsoluteCenter(ix);
                Vec3 pos(x, y, z);
                Vec3 pos_original = rot_inv * (pos - trans);
                Index ix_original = input_shape.VoxelIndex(pos_original);
                if (input_shape.InRange(ix_original)) {
                    sum += input[input_shape.at(ix_original)];
                }
                else {
                    sum += defvar;
                }
            }
        }
    }
    output[output_shape.at(ix)] = sum / (super * super * super);

}


HostVolume<uint16_t>
cuda_affineinv(
const HostVolume<uint16_t> &input,
const HostVolume<uint16_t> &output_shape,
Mat3 rot_inv, Vec3 trans, uint16_t defvar)
{
    DeviceVolume<uint16_t> input_dev = input;
    DeviceVolume<uint16_t> output_dev;
    output_dev.CopyShape(output_shape);
    size_t x = output_dev.Nx();
    size_t y = output_dev.Ny();
    size_t z = output_dev.Nz();
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocks(x / threadsPerBlock.x + 1, y / threadsPerBlock.y + 1, z / threadsPerBlock.z + 1);
    affineinv_kernel<<<blocks, threadsPerBlock>>>
        (input_dev.data(), input_dev.GetShape(), output_dev.data(), output_dev.GetShape(), rot_inv, trans, defvar);
    HostVolume<uint16_t> output_host = output_dev;
    return std::move(output_host);
}

__global__ void shrink_ss_kernel(uint16_t *output, Shape output_shape, const uint16_t *supersampled, Shape supersampled_shape, size_t ss)
{
    int vix_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vix_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vix_z = blockIdx.z * blockDim.z + threadIdx.z;
    Index ix(vix_x, vix_y, vix_z);
    if (!output_shape.InRange(ix)) {
        return;
    }
    uint32_t sum = 0;
    for (int i = 0; i < ss; i++) {
        for (int j = 0; j < ss; j++) {
            for (int k = 0; k < ss; k++) {
                size_t ixs_ss = supersampled_shape.at(ix.x + i, ix.y + j, ix.z + k);
                sum += supersampled[ixs_ss];
            }
        }
    }
    output[output_shape.at(ix)] = sum / (ss * ss * ss);
    
}

HostVolume<uint16_t>
cuda_affineinv(
const HostVolume<uint16_t> &input,
const HostVolume<uint16_t> &output_shape,
Mat3 rot_inv, Vec3 trans, uint16_t defvar, size_t super)
{
    if (super <= 1) {
        return cuda_affineinv(input, output_shape, rot_inv, trans, defvar);
    }
    DeviceVolume<uint16_t> input_dev = input;
    DeviceVolume<uint16_t> output_dev;
    output_dev.CopyShape(output_shape);
    size_t x = output_dev.Nx();
    size_t y = output_dev.Ny();
    size_t z = output_dev.Nz();
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocks(x / threadsPerBlock.x + 1, y / threadsPerBlock.y + 1, z / threadsPerBlock.z + 1);
    affineinv_kernel <<<blocks, threadsPerBlock >>>
        (input_dev.data(), input_dev.GetShape(), output_dev.data(), output_dev.GetShape(), rot_inv, trans, defvar, super);
    cudaDeviceSynchronize();
    HostVolume<uint16_t> output_host = output_dev;
    return std::move(output_host);
}