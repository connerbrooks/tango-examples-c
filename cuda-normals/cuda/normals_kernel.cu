////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#ifndef _NORMALS_KERNEL_CU_
#define _NORMALS_KERNEL_CU_

#include "normals_kernel.h"
#include "helper_math.h"
#include "helper_functions.h"

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// texture memory is used to store the image data
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
cudaArray *d_array;

namespace gpu_bf
{
    __global__ void d_compute_normals(const float *h_depth, float *h_normals, int *indices, int h_depth_buffer_size)
    {
        int i = blockIdx.x + threadIdx.x;
        
        int idx = i * 3;
        int nnIdx = i * 4; 

        int n1 = indices[nnIdx + 1];
        int n2 = indices[nnIdx + 2];

        // get points
        float3 point_interest = make_float3(h_depth[idx], h_depth[idx+1], h_depth[idx+2]);
        float3 neighbor_1 = make_float3(h_depth[n1], h_depth[n1+1], h_depth[n1+2]);
        float3 neighbor_2 = make_float3(h_depth[n2], h_depth[n2+1], h_depth[n2+2]);

        float3 vec1 = point_interest - neighbor_1;
        float3 vec2 = point_interest - neighbor_2;

        float3 vec_norm = cross(vec1, vec2);

        if(dot(point_interest, vec_norm) > 0)
          vec_norm = -vec_norm;

        h_normals[idx] = vec_norm.x;
        h_normals[idx + 1] = vec_norm.y;
        h_normals[idx + 2] = vec_norm.z;
      
    }

    Normals::Normals()
    {
    }

    // free the allocated memory
    Normals::~Normals()
    {
        checkCudaErrors(cudaFree(d_depth));
        checkCudaErrors(cudaFree(d_normals));
        checkCudaErrors(cudaFree(d_indices));
    }

    // initialize the texture with the input image array
    void Normals::init(float* h_depth, int *h_indices, int depth_buffer_size, int k) 
    {
        // allocate memory to the depth, normals, and indicies 
        int size = depth_buffer_size * sizeof(float);
        checkCudaErrors(cudaMalloc(&d_depth, size * 3));    // verticies * xyz
        checkCudaErrors(cudaMalloc(&d_normals, size * 3)); 
        checkCudaErrors(cudaMalloc(&d_indices, size * k));  // size * number of neighbors


        // copy depth data to device
        checkCudaErrors(cudaMemcpy(d_depth, h_depth, size * 3, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_indices, h_indices, size * k, cudaMemcpyHostToDevice));
    }

    // apply Box filter
    void Normals::calculateNormals(const float *h_depth, float *h_normals, int *h_indices, int h_depth_buffer_size )
    {

        // sync host and start kernel computation timer_kernel
        checkCudaErrors(cudaDeviceSynchronize());


        // blocks / threads
        //dim3 threadsPerBlock(16, 16);
        //dim3 numBlocks(h_depth_buffer_size / threadsPerBlock.x, h_depth_buffer_size / threadsPerBlock.y);

        int threadsPerBlock = 256;
        int numBlocks = h_depth_buffer_size / threadsPerBlock;

        // call kernel
        d_compute_normals<<<numBlocks, threadsPerBlock>>>(d_depth, d_normals, d_indices, h_depth_buffer_size);

        // copy normals buffer back to host
        checkCudaErrors(cudaMemcpy(h_normals, d_normals,  h_depth_buffer_size * 3  * sizeof(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaDeviceSynchronize());
    }
}
#endif // #ifndef _NORMALS_KERNEL_CU_
