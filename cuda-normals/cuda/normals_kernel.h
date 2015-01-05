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
            // initialization of the textures here
            void init(int width, int height,const void *pImage);
            // box filter is applied here
            void calculateNormals(unsigned int *h_dest, int width, int height, int radius, int iterations, int nthreads);
        private:
            // temp array for storing intermediate result
            unsigned int *d_temp;
            // final result of applying a box filter
            unsigned int *d_result;
    };
}//namespace gpu_bf

#endif

