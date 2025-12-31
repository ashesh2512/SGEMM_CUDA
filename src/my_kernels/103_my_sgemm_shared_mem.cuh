#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

template <const int tileDim>
__global__ void my_sgemm_shared_mem(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C)
{
  // in this example, the dimension of shared memory is the size of block itself - (4+4)kB per block
  __shared__ float A_s[tileDim][tileDim];
  __shared__ float B_s[tileDim][tileDim];

  // NEVER conflate row and column with x and y axis of the thread block
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < M && col < N)
  {
    float tmp = 0.0;
    for(uint tile=0; tile < K/tileDim; ++tile)
    {
      // access of A and B is row major format
      // one way to think of the A and B indexing is find the starting point for each tile
      A_s[threadIdx.y][threadIdx.x] = A[row*K + tile*tileDim + threadIdx.x];
      B_s[threadIdx.y][threadIdx.x] = B[(tile*tileDim + threadIdx.y)*N + col];
      __syncthreads();

      for(uint i = 0; i < tileDim; ++i)
      {
        // tmp accumulates the tile-wise contirbution to C(row, col)
        // at the end of the day, each thread contributes to one entry in C
        tmp += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
      }
      __syncthreads();
    }

    // C = α*(A*B)+β*C
    C[row*N + col] = alpha*tmp + beta * C[row*N + col];
  }
}