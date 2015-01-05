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
    // reads from 32-bit unsigned int array holding 8-bit RGBA

    // convert floating point rgba color to 32-bit integer
    __device__ unsigned int rgbaFloatToInt(float4 rgba)
    {
        rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
        rgba.y = __saturatef(rgba.y);
        rgba.z = __saturatef(rgba.z);
        rgba.w = __saturatef(rgba.w);
        return ((unsigned int)(rgba.w * 255.0f) << 24) |
               ((unsigned int)(rgba.z * 255.0f) << 16) |
               ((unsigned int)(rgba.y * 255.0f) <<  8) |
               ((unsigned int)(rgba.x * 255.0f));
    }

    __device__ float4 rgbaIntToFloat(unsigned int c)
    {
        float4 rgba;
        rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
        rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
        rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
        rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
        return rgba;
    }

    // row pass using texture memory reads
    __global__ void d_boxfilter_rgba_x(unsigned int *od, int w, int h, int r)
    {
        float scale = 1.0f / (float)((r << 1) + 1);
        unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

        // as long as address is always less than height, we do work
        if (y < h)
        {
            float4 t = make_float4(0.0f);

            for (int x = -r; x <= r; x++)
            {
                t += tex2D(rgbaTex, x, y);
            }

            od[y * w] = rgbaFloatToInt(t * scale);

            for (int x = 1; x < w; x++)
            {
                t += tex2D(rgbaTex, x + r, y);
                t -= tex2D(rgbaTex, x - r - 1, y);
                od[y * w + x] = rgbaFloatToInt(t * scale);
            }
        }
    }

    // column pass using coalesced global memory reads
    __global__ void d_boxfilter_rgba_y(unsigned int *id, unsigned int *od, int w, int h, int r)
    {
        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        id = &id[x];
        od = &od[x];

        float scale = 1.0f / (float)((r << 1) + 1);

        float4 t;
        // do left edge
        t = rgbaIntToFloat(id[0]) * r;

        for (int y = 0; y < (r + 1); y++)
        {
            t += rgbaIntToFloat(id[y*w]);
        }

        od[0] = rgbaFloatToInt(t * scale);

        for (int y = 1; y < (r + 1); y++)
        {
            t += rgbaIntToFloat(id[(y + r) * w]);
            t -= rgbaIntToFloat(id[0]);
            od[y * w] = rgbaFloatToInt(t * scale);
        }

        // main loop
        for (int y = (r + 1); y < (h - r); y++)
        {
            t += rgbaIntToFloat(id[(y + r) * w]);
            t -= rgbaIntToFloat(id[((y - r) * w) - w]);
            od[y * w] = rgbaFloatToInt(t * scale);
        }

        // do right edge
        for (int y = h - r; y < h; y++)
        {
            t += rgbaIntToFloat(id[(h - 1) * w]);
            t -= rgbaIntToFloat(id[((y - r) * w) - w]);
            od[y * w] = rgbaFloatToInt(t * scale);
        }
    }

    Normals::Normals()
    {
    }

    // free the allocated memory
    Normals::~Normals()
    {
        checkCudaErrors(cudaFree(d_temp));
        checkCudaErrors(cudaFree(d_result));
    }

    // initialize the texture with the input image array
    void Normals::init(int width, int height,const void *pImage)
    {
        // allocate memory to the intermediate arrays
        int size = width * height * sizeof(unsigned int);
        checkCudaErrors(cudaMalloc(&d_temp, size));
        checkCudaErrors(cudaMalloc(&d_result, size));

        // copy image data to array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        checkCudaErrors(cudaMallocArray(&d_array, &channelDesc, width, height));
        checkCudaErrors(cudaMemcpyToArray(d_array, 0, 0, pImage, size, cudaMemcpyHostToDevice));
    }

    // apply Box filter
    void Normals::calculateNormals(unsigned int *h_dest, int width, int height, int radius, int iterations, int nthreads)
    {
        // bind texture to the array containing input image data
        checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_array));

        for (int i=0; i<iterations; i++)
        {
            // sync host and start kernel computation timer_kernel
            checkCudaErrors(cudaDeviceSynchronize());

            // use texture for horizontal pass
            d_boxfilter_rgba_x<<< height / nthreads, nthreads, 0 >>>(d_temp, width, height, radius);
            // use array for vertical paas
            d_boxfilter_rgba_y<<< width / nthreads, nthreads, 0 >>>(d_temp, d_result, width, height, radius);

            // sync host and stop computation timer_kernel
            checkCudaErrors(cudaDeviceSynchronize());

            // copy back to the host memory
            checkCudaErrors(cudaMemcpy(h_dest, d_result,  width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
    }
}
#endif // #ifndef _BOXFILTER_KERNEL_CU_
