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

#ifndef _NORMALS_KERNEL_H_
#define _NORMALS_KERNEL_H_

namespace gpu_bf
{
    class Normals 
    {
        public:
            Normals();
            ~Normals();
            // initialization of buffers 
            void init(float *h_depth, int *h_indices, int depth_buffer_size, int k);
            // normals are calculated here
            void calculateNormals(const float *h_normals, float *h_depth, int *h_indices, int h_depth_buffer_size);
        private:
            // temp array for storing intermediate result
            float *d_depth;
            // final result for normals 
            float *d_normals;
            // knn indices
            int *d_indices;
            // depth buffer size
            int depth_buffer_size;
            int k;

    };
}//namespace gpu_bf

#endif

