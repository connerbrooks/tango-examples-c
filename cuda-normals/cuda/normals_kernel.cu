
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

namespace gpu_bf
{
	__global__ void d_compute_normals(float *d_depth, float *d_normals, float **d_projection_matrix, 
									  float **d_ij_matrix, int d_screen_width, int d_screen_height) 
	{
        int thread_iter = blockIdx.x + threadIdx.x;
        int vertex_iter = thread_iter * 3;
		
		// this 3d coordinate needs to be converted into clip space.	
		float3 point_of_interest = /* viewProjectionMatrix  * */ make_float3(d_depth[vertex_iter], d_depth[vertex_iter+1], d_depth[vertex_iter+2]);	

		// adding by 0.5 and casting to an int acts as a rounding function.	
		int i = (int)((((point_of_interest[0] + 1) / 2) * d_screen_width) + 0.5);
		int j = (int)((((1 - point_of_interest[1]) / 2) * d_screen_height) + 0.5);
		
		// store index of the x component of the 3d point of interest into the ij matrix.
		d_ij_matrix[i][j] = thread_iter;

		// let all threads fill the ij matrix.	
		__syncthreads();

		int neighbor_1_index, neighbor_2_index;
		float3 neighbor_1, neighbor_2;
		if (i > 0) {	// don't let it go out of bounds.	
			neighbor_1_index = d_ij_matrix[i-1][j];	
			neighbor_1 = make_float3(d_depth[neighbor_1_index], d_depth[neighbor_1_index+1], d_depth[neighbor_1_index+2]);
		}
		if (j < d_screen_height) {
			neighbor_2_index = d_ij_matrix[i][j+1];
			neighbor_2 = make_float3(d_depth[neighbor_2_index], d_depth[neighbor_2_index+1], d_depth[neighbor_2_index+2]); 
		}

		// do normal calculation.
		float3 vec1 = neighbor_1 - point_of_interest;	
		float3 vec2 = neighbor_2 - point_of_interest;	
		float3 normal = normalize(cross(vec1, vec2));

		// make sure it's facing the viewport.
		if(dot(point_of_interest, normal) > 0)
        	normal = -normal;
	
		// store the values.
        d_normals[vertex_iter]     = normal.x;
        d_normals[vertex_iter + 1] = normal.y;
        d_normals[vertex_iter + 2] = normal.z;
	}

    Normals::Normals()
    {
    }

    // free the allocated memory
    Normals::~Normals()
    {
		checkCudaErrors(cudaFree(d_screen_width));
		checkCudaErrors(cudaFree(d_screen_height));
        checkCudaErrors(cudaFree(d_depth));
        checkCudaErrors(cudaFree(d_normals));

		for (int i = 0; i < c_screen_width; i++) {
			checkCudaErrors(cudaFree(d_ij_matrix[i]));
		}
        checkCudaErrors(cudaFree(d_ij_matrix));

		for (int j = 0; j < 4; j++) {
			checkCudaErrors(cudaFree(d_projection_matrix[j]));
		}
        checkCudaErrors(cudaFree(d_projection_matrix));
    }

    // initialize the ij matrix.
    void Normals::calculateNormals(float* h_depth, float *h_normals, float **h_projection_matrix, 
								   int h_depth_buffer_size, int h_screen_width, int h_screen_height) 
    {
		////////////////////////////////////////////////////////////////////////////// initialize memory

        int size = h_depth_buffer_size * sizeof(float) * 3;
        
		checkCudaErrors(cudaMalloc(&d_normals, size));    // verticies * xyz
		
		// allocate memory to the ij matrix.
		checkCudaErrors(cudaMalloc((void**)&d_ij_matrix, h_screen_width * sizeof(float*)));
		for (int i = 0; i < h_screen_width; i++) {
			checkCudaErrors(cudaMalloc(&d_ij_matrix[i], h_screen_height * sizeof(float))); 
			checkCudaErrors(cudaMemset((void*)d_ij_matrix[i], -1, h_screen_height * sizeof(float));
		}

		// allocate projection matrix memory and copy to cuda matrix.
		checkCudaErrors(cudaMalloc((void**)&d_projection_matrix, 4 * sizeof(float*)));
		for (int i = 0; i < 4; i++) {
			checkCudaErrors(cudaMalloc(&d_projection_matrix[i], 4 * sizeof(float))); 
			checkCudaErrors(cudaMemcpy(d_projection_matrix[i], &h_projection_matrix[i], 
							4 * sizeof(float), cudaMemcpyHostToDevice));
		}
	
		// allocate depth memory and copy to cuda array.
		checkCudaErrors(cudaMalloc(&d_depth, size));    // verticies * xyz
        checkCudaErrors(cudaMemcpy(d_depth, h_depth, size, cudaMemcpyHostToDevice));
	
		// allocate screen width/height memory and copy to cuda array.
		checkCudaErrors(cudaMalloc((void**)&d_screen_width, sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_screen_width, &h_screen_width, sizeof(int), cudaMemcpyHostToDevice));
		
		checkCudaErrors(cudaMalloc((void**)&d_screen_height, sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_screen_height, &h_screen_height, sizeof(int), cudaMemcpyHostToDevice));

		//////////////////////////////////////////////////////////////////////////////////// call kernal

        // sync host and start kernel computation timer_kernel
		checkCudaErrors(cudaDeviceSynchronize());

		int threadsPerBlock = 256;
        int numBlocks = h_depth_buffer_size / threadsPerBlock;
		
		// call kernel
		d_compute_normals<<<numBlocks, threadsPerBlock>>>(d_depth, d_normals, d_projection_matrix, 
														  d_ij_matrix, h_screen_width, h_screen_height);

		/////////////////////////////////////////////////////////////////////// copy memory back to host

        checkCudaErrors(cudaMemcpy(h_normals, d_normals, size, cudaMemcpyDeviceToHost));
        
		checkCudaErrors(cudaDeviceSynchronize());
    }
}
#endif // #ifndef _NORMALS_KERNEL_CU_
