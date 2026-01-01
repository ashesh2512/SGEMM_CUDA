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
template <const int BM, const int BN, const int BK, const int TM>
__global__ void my_sgemm_1D_multi_elems(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C)
{
  __shared__ float A_s[BM][BK];
  __shared__ float B_s[BK][BN];

  // NEVER conflate row and column with x and y axis of the thread block
  const uint C_block_row = blockIdx.y * BM;
  const uint C_block_col = blockIdx.x * BN;

  // chunk rows and columns
  const uint A_chunk_col = threadIdx.x % BK; // 0 to 7
  const uint A_chunk_row = threadIdx.x / BK; // 0 to 63
  const uint B_chunk_col = threadIdx.x % BN; // 0 to 63
  const uint B_chunk_row = threadIdx.x / BN; // 0 to 7

  // we call this thread, and not chunk, because the C chunk (64 x 64) is bigger than the number of threads in the block
  const uint C_thread_col = threadIdx.x % BN; // 0 to 63
  const uint C_thread_row = threadIdx.x / BN; // 0 to 7 - each thread contributes to TM (8) values in C

  float tmp[TM] = {0.0f}; // TM values per thread

  for(uint tile=0; tile < K/BK; ++tile)
  {
    uint global_A_row = C_block_row + A_chunk_row; // accounts for rows processed by previous blocks
    uint global_A_col = tile*BK + A_chunk_col; // in a given row, no column is processed by a previous block
    A_s[A_chunk_row][A_chunk_col] = A[global_A_row*K + global_A_col];

    uint global_B_row = tile*BK + B_chunk_row;// in a given col, no row is processed by a previous block
    uint global_B_col = C_block_col + B_chunk_col; // accounts for cols processed by previous blocks
    B_s[B_chunk_row][B_chunk_col] = B[global_B_row*N + global_B_col];
    __syncthreads();

    for (uint i = 0; i < BK; ++i) {
      // this is basically a dotproduct
      float b_val = B_s[i][C_thread_col];
      for (uint t = 0; t < TM; ++t) {
        tmp[t] += A_s[(C_thread_row * TM + t)][i] * b_val;
      }
    }
    __syncthreads();
  }

  // C = α*(A*B)+β*C
  for (uint t = 0; t < TM; ++t) {
    const uint base_row = C_block_row + C_thread_row * TM + t;
    const uint base_col = C_block_col + C_thread_col;
    C[base_row * N + base_col] =
        alpha * tmp[t] + beta * C[base_row * N + base_col];
  }
}


