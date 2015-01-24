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
            void calculateNormals(float *h_depth, float *h_normals, int h_depth_buffer_size, int h_screen_width, int h_screen_height);
        private:
            // temp array for storing depth information.
            float *d_depth;
			float *d_normals;
   
	         // stores indices of 3d points in a 2d array with dimensions being their screen coordinates.
            float **d_ij_matrix;
			// stores projection matrix to convert buffer coordinates to clip space.
			float **d_projection_matrix;

			int d_screen_width;
			int d_screen_height;
	};
}//namespace gpu_bf

#endif

