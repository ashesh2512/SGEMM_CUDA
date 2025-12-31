#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void my_sgemm_global_coalesced(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C)
{
  // NEVER conflate row and column with x and y axis of the thread block
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if(row < M && col < N)
  {
    float tmp = 0.0;
    for (int k = 0; k < K; ++k)
    {
      // A=M(size of row) x K(size of column)
      // B=K(size of row) x N(size of column)
      tmp += A[row*K + k] // all threads in the warp are accessing the same entry - this is a broadcast within a warp
           * B[k*N + col]; // this is coalesced
    }
    // C = α*(A@B)+β*C
    C[row*N + col] = alpha*tmp + beta * C[row*N + col];
  }
}