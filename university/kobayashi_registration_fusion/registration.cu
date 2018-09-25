
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include "registration_kernel.h"

__global__ void transform_points_kernel(const Vec4 *input, Vec3 *output, size_t n, Mat3 rotation, Vec3 translation)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < n) {
        const Vec4 &v4 = input[ix];
        output[ix] = rotation * Vec3(v4.x, v4.y, v4.z) + translation;
    }
}


std::vector<Vec3> cuda_transform_points(const std::vector<Vec4> &points, const Mat3 &rotation, const Vec3 &translation)
{
    thrust::device_vector<Vec4> points_dev(points.size());
    thrust::copy(points.begin(), points.end(), points_dev.begin());
    thrust::device_vector<Vec3> result_dev(points.size());
    size_t threadsPerBlock = 256;
    size_t blockdim = points.size() / threadsPerBlock + 1;
    transform_points_kernel<<<blockdim,threadsPerBlock>>>(
        thrust::raw_pointer_cast(points_dev.data()), thrust::raw_pointer_cast(result_dev.data()), points.size(), rotation, translation);
    std::vector<Vec3> result(points.size());
    thrust::copy(result_dev.begin(), result_dev.end(), result.begin());
    return std::move(result);
}

__global__ void associate_closest_kernel(thrust::device_ptr<Vec4> orig_points, thrust::device_ptr<Vec3> moved_points, thrust::device_ptr<Vec3> result, size_t n, int root, FDT<DeviceKind::Cuda>::vector_type::const_iterator head)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < n) {
        float w = orig_points.get()[ix].w;
        Vec3 pos = moved_points.get()[ix];
        Vec4 q ( pos.x, pos.y, pos.z, w );
        Vec4 nearest = FDT<DeviceKind::Cuda>::nearest(q, root, head);
        result.get()[ix] = { nearest.x, nearest.y, nearest.z };
    }
}

__global__ void associate_closest_kernel_single(thrust::device_ptr<Vec4> orig_points, thrust::device_ptr<Vec3> moved_points, thrust::device_ptr<Vec3> result, size_t n, int root, FDT<DeviceKind::Cuda>::vector_type::const_iterator head, size_t ix)
{
    if (ix < n) {
        float w = orig_points.get()[ix].w;
        Vec3 pos = moved_points.get()[ix];
        Vec4 q(pos.x, pos.y, pos.z, w);
        Vec4 nearest = FDT<DeviceKind::Cuda>::nearest(q, root, head);
        result.get()[ix] = { nearest.x, nearest.y, nearest.z };
    }
}

__global__ void associate_closest_kernel_linear(
    Vec4 *orig_points, Vec3 *moved_points, Vec3 *result, size_t tar_size, 
    Vec4 *ref_points, size_t ref_size)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < tar_size) {
        float w = orig_points[ix].w;
        Vec3 p = moved_points[ix];
        Vec4 v = { p.x, p.y, p.z, w };
        Vec4 best;
        float best_err = -1;
        for (size_t i = 0; i < ref_size; i++) {
            Vec4 u =  ref_points[i];
            float ds[4] = { u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w };
            float err = 0;
            for (int i = 0; i < 4; i++) { err += ds[i] * ds[i]; }
            if (best_err < 0 || err < best_err) {
                best_err = err;
                best = u;
            }
        }
        result[ix] = { best.x, best.y, best.z };
    }
}

__global__ void sanity_check(FDT<DeviceKind::Cuda>::vector_type::const_iterator head, size_t size) {
    printf("sanity check %lld\n", size);
    for (size_t i = 0; i < size; i++) {
        const FDTNode &node = head[i];
        if (node.lesser >= (int64_t)size || node.greater >= (int64_t)size) {
            printf("oor %lld %d %d\n", i, node.lesser, node.greater);
        }
    }
}
std::vector<Vec3> cuda_associate_closest(const std::vector<Vec4> &orig_points, const std::vector<Vec3> moved_points, const std::vector<Vec4> &ref_points)
{
    cudaSetDevice(1);
    size_t threadsPerBlock = 256;

    Vec3 *result_dev;
    cudaMalloc(&result_dev, sizeof(Vec3)* orig_points.size());
    {
        Vec4 *orig_points_dev;
        cudaMalloc(&orig_points_dev, sizeof(Vec4)* orig_points.size());
        cudaMemcpy(orig_points_dev, orig_points.data(), sizeof(Vec4)* orig_points.size(), cudaMemcpyHostToDevice);

        Vec3 *moved_points_dev;
        cudaMalloc(&moved_points_dev, sizeof(Vec3)* moved_points.size());
        cudaMemcpy(moved_points_dev, moved_points.data(), sizeof(Vec3)* moved_points.size(), cudaMemcpyHostToDevice);

        Vec4 *ref_points_dev;
        cudaMalloc(&ref_points_dev, sizeof(Vec4)* ref_points.size());
        cudaMemcpy(ref_points_dev, ref_points.data(), sizeof(Vec4)* ref_points.size(), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        size_t chunksize = 256 * 256;
        for (int i = 0; i < orig_points.size(); i += chunksize) {
            size_t size = orig_points.size() - i;
            if (size > chunksize) { size = chunksize; }
            size_t blockdim = chunksize / threadsPerBlock + 1;

            associate_closest_kernel_linear << <blockdim, threadsPerBlock >> >(
                orig_points_dev + i, moved_points_dev + i, result_dev + i, size, ref_points_dev, ref_points.size());
            cudaDeviceSynchronize();
        }
        cudaFree(orig_points_dev);
        cudaFree(moved_points_dev);
        cudaFree(ref_points_dev);
    }

    std::vector<Vec3> result(orig_points.size());
    cudaMemcpy(result.data(), result_dev, sizeof(Vec3)* orig_points.size(), cudaMemcpyDeviceToHost);
    cudaFree(result_dev);
    return std::move(result);

}