
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernel.h"
#include "volume.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>




__global__ void laplacian_kernel(
    const uint16_t *input,
    uint16_t *output,
    Shape shape)
{
    int vix_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vix_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vix_z = blockIdx.z * blockDim.z + threadIdx.z;
    Index ix(vix_x, vix_y, vix_z);
    if (!shape.InRange(ix)) {
        return;
    }

    int cnt = 0;
    int32_t val = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            for (int k = -1; k < 2; k++) {
                Index ix_here(vix_x + i, vix_y + j, vix_z + k);
                if (shape.InRange(ix_here)) {
                    cnt++;
                    val -= input[shape.at(ix_here)];
                }
            }
        }
    }
    val += cnt * input[shape.at(ix)];
    if (val < 0) {
        val = -val;
    }
    if (val > 0xffff) {
        val = 0xffff;
    }
   
    uint16_t v = val & 0xffff;
    output[shape.at(ix)] = v;
}

__global__ void filter_kernel(
    uint16_t *volume, Shape volume_shape, uint16_t *eval, Shape eval_shape,
    int32_t ev_min, int32_t ev_max)
{
    int vix_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vix_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vix_z = blockIdx.z * blockDim.z + threadIdx.z;
    Index ix(vix_x, vix_y, vix_z);
    if (!volume_shape.InRange(ix)) {
        return;
    }
    Vec3 pos = volume_shape.AbsoluteCenter(ix);
    Index ix_ev = eval_shape.VoxelIndex(pos);
    if (eval_shape.InRange(ix_ev)) {
        int32_t v = volume[volume_shape.at(ix)];
        int32_t ev = eval[eval_shape.at(ix_ev)];
        if (ev < ev_min || ev_max < ev) {
            volume[volume_shape.at(ix)] = 0;
        }
    }
    else {
        volume[volume_shape.at(ix)] = 0;
    }
}

__global__ void calc_pos(
    const uint16_t *volume, Shape volume_shape, Vec4 *output)
{
    int vix_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vix_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vix_z = blockIdx.z * blockDim.z + threadIdx.z;
    Index ix(vix_x, vix_y, vix_z);
    if (!volume_shape.InRange(ix)) {
        return;
    }
    Vec3 pos = volume_shape.AbsoluteCenter(ix);
    Vec4 v4(pos.x, pos.y, pos.z, (float)volume[volume_shape.at(ix)]);
    output[volume_shape.at(ix)] = v4;
}




VolumeCoord<uint16_t, DeviceKind::Host> cuda_laplacian(const VolumeCoord<uint16_t, DeviceKind::Host> &input_host) {
   
    VolumeCoord<uint16_t, DeviceKind::Cuda> input_dev = input_host;
   
    VolumeCoord<uint16_t, DeviceKind::Cuda> output_dev;
  
    output_dev.CopyShape(input_dev);
    size_t x = input_dev.Nx();
    size_t y = input_dev.Ny();
    size_t z = input_dev.Nz();
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocks(x / threadsPerBlock.x + 1, y / threadsPerBlock.y + 1, z / threadsPerBlock.z + 1);
    laplacian_kernel<<<blocks, threadsPerBlock >>>(input_dev.data(), output_dev.data(), input_dev.GetShape());
   
    VolumeCoord<uint16_t, DeviceKind::Host> output_host = output_dev;
   
    return std::move(output_host);
}


HostVolume<uint16_t>
cuda_filtervolume(
    const HostVolume<uint16_t> &orig,
    const HostVolume<uint16_t> &vol, int32_t vol_min,
    const HostVolume<uint16_t> &lap, int32_t lap_min,
    const HostVolume<uint16_t> &aff, int32_t aff_max)
{
    size_t x = orig.Nx();
    size_t y = orig.Ny();
    size_t z = orig.Nz();
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocks(x / threadsPerBlock.x + 1, y / threadsPerBlock.y + 1, z / threadsPerBlock.z + 1);

    DeviceVolume<uint16_t> filtered_dev = orig;
    DeviceVolume<uint16_t> vol_dev = vol;
    filter_kernel <<<blocks, threadsPerBlock>>>(
        filtered_dev.data(), filtered_dev.GetShape(), 
        vol_dev.data(), vol_dev.GetShape(), vol_min, 0xFFFF);
    DeviceVolume<uint16_t> lap_dev = lap;
    filter_kernel <<<blocks, threadsPerBlock>>>(
        filtered_dev.data(), filtered_dev.GetShape(),
        lap_dev.data(), lap_dev.GetShape(), lap_min, 0xFFFF);
    DeviceVolume<uint16_t> aff_dev = aff;
    filter_kernel <<<blocks, threadsPerBlock>>>(
        filtered_dev.data(), filtered_dev.GetShape(), 
        aff_dev.data(), aff_dev.GetShape(), 0, aff_max);

    HostVolume<uint16_t> filtered_host = filtered_dev;
    return std::move(filtered_host);
}

struct WNeg : public thrust::unary_function<Vec4, bool> {
    DEVICE_HOST bool operator() (const Vec4 &v) {
        return v.w <= 0;
    }
};
std::vector<Vec4> cuda_samplepoints
(
const HostVolume<uint16_t> &orig,
const HostVolume<uint16_t> &vol, int32_t vol_min,
const HostVolume<uint16_t> &lap, int32_t lap_min,
const HostVolume<uint16_t> &aff, int32_t aff_max)
{
    size_t x = orig.Nx();
    size_t y = orig.Ny();
    size_t z = orig.Nz();
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocks(x / threadsPerBlock.x + 1, y / threadsPerBlock.y + 1, z / threadsPerBlock.z + 1);

    DeviceVolume<uint16_t> filtered_dev = orig;
    {
        DeviceVolume<uint16_t> vol_dev = vol;
        filter_kernel<<<blocks, threadsPerBlock>>>(
            filtered_dev.data(), filtered_dev.GetShape(),
            vol_dev.data(), vol_dev.GetShape(), vol_min, 0xFFFF);
        DeviceVolume<uint16_t> lap_dev = lap;
        filter_kernel<<<blocks, threadsPerBlock>>>(
            filtered_dev.data(), filtered_dev.GetShape(),
            lap_dev.data(), lap_dev.GetShape(), lap_min, 0xFFFF);
        DeviceVolume<uint16_t> aff_dev = aff;
        filter_kernel<<<blocks, threadsPerBlock>>>(
            filtered_dev.data(), filtered_dev.GetShape(),
            aff_dev.data(), aff_dev.GetShape(), 0, aff_max);
    }
    std::cout << "filter completed "  << x*y*z << std::endl;
    thrust::device_vector<Vec4> points(x * y * z);

    calc_pos << <blocks, threadsPerBlock >> >(
        filtered_dev.data(), filtered_dev.GetShape(), thrust::raw_pointer_cast(points.data()));
    cudaDeviceSynchronize();
    WNeg wneg;
    std::vector<Vec4> points_host(points.size());
    thrust::copy(points.begin(), points.end(), points_host.begin());
    auto end = std::remove_if(points_host.begin(), points_host.end(), wneg);
    int num = std::distance(points_host.begin(), end);
    points_host.resize(num);
    return points_host;
    
    /*
    auto end = thrust::remove_if(points.begin(), points.end(), wneg);

    int num = thrust::distance(points.begin(), end);

    std::cout << __LINE__ << std::endl;
    std::vector<Vec4> points_host(num);

    std::cout << __LINE__ << std::endl;
    thrust::copy(points.begin(), end, points_host.begin());

    std::cout << __LINE__ << std::endl;
    return points_host;
    */
  

}
