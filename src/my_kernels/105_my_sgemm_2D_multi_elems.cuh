#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

// lot of the logic below assumes everything is a multiple of everything
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void my_sgemm_2D_multi_elems(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C)
{
    // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numBlockThreads = (BM * BN) / (TM * TN);
  assert(numBlockThreads == blockDim.x);
  
  __shared__ float A_s[BM][BK]; // 128x8 here
  __shared__ float B_s[BK][BN]; // 8x128 here

  // base value for current block of rows and columns for 128x128 C block being populated
  const uint C_block_row = blockIdx.y * BM;
  const uint C_block_col = blockIdx.x * BN;

  // chunk rows and columns
  // threadIdx.x varies from 0 to 255
  const uint A_chunk_row = threadIdx.x / BK; // 0 to 31
  const uint A_chunk_col = threadIdx.x % BK; // 0 to 7
  const uint B_chunk_row = threadIdx.x / BN; // 0 to 1
  const uint B_chunk_col = threadIdx.x % BN; // 0 to 127

  const uint strideA = numBlockThreads / BK;
  const uint strideB = numBlockThreads / BN;

  // we call this thread, and not chunk, because the C chunk (128 x 128) is bigger than the number of threads in the block
  const uint C_col_thread = threadIdx.x % (BN/TN); // 0 to 15 - each thread contributes to (8) col indices in C
  const uint C_row_thread = threadIdx.x / (BN/TN); // 0 to 15 - each thread contributes to (8) row indices in C

  float tmp[TM][TN] = {0.0f}; // TMxTN values per thread
  float regM[TM] = {0.0f};
  float regN[TM] = {0.0f};

  for(uint tile=0; tile < K; tile += BK)
  {
    for (uint offset = 0; offset < BM; offset += strideA) {
      A_s[A_chunk_row + offset][A_chunk_col] =
          A[(C_block_row + A_chunk_row + offset) * K + tile + A_chunk_col];
    }

    for (uint offset = 0; offset < BK; offset += strideB) {
      B_s[B_chunk_row + offset][B_chunk_col] =
          B[(B_chunk_row + offset + tile) * N + C_block_col + B_chunk_col];
    }
    __syncthreads();

    for (uint i = 0; i < BK; ++i) {

      for (uint tm = 0; tm < TM; ++tm)
      {
        regM[tm] = A_s[C_row_thread * TM + tm][i];
      }
      for (uint tn = 0; tn < TN; ++tn)
      {
        regN[tn] = B_s[i][C_col_thread * TN + tn];
      }

      for (uint tm = 0; tm < TM; ++tm) {
        for (uint tn = 0; tn < TN; ++tn) {
          tmp[tm][tn] += regM[tm]*regN[tn];
        }
      }
    }
    __syncthreads();
  }

  // C = α*(A*B)+β*C
  for (uint tm = 0; tm < TM; ++tm)
  {
    const uint base_row = C_block_row + C_row_thread * TM + tm;
    for (uint tn = 0; tn < TN; ++tn)
    {
      const uint base_col = C_block_col + C_col_thread * TN + tn;
      C[base_row * N + base_col] =
          alpha * tmp[tm][tn] + beta * C[base_row * N + base_col];
    }
  }
}
