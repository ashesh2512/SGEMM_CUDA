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
__global__ void my_sgemm_vectorize(int M, int N, int K, float alpha, float *A,
                                   float *B, float beta, float *C)
{
    // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numBlockThreads = (BM * BN) / (TM * TN);
  assert(numBlockThreads == blockDim.x);
  
  __shared__ float A_s[BK][BM]; // 8x128 here - transposed for easy vectorization
  __shared__ float B_s[BK][BN]; // 8x128 here

  // base value for current block of rows and columns for 128x128 C block being populated
  const uint C_block_row = blockIdx.y * BM;
  const uint C_block_col = blockIdx.x * BN;

  // chunk rows and columns
  // threadIdx.x varies from 0 to 255
  // load 128bit / 32bit = 4 elements per thread at each tile
  const uint A_chunk_row = threadIdx.x / (BK/4);
  const uint A_chunk_col = threadIdx.x % (BK/4);
  const uint B_chunk_row = threadIdx.x / (BN/4);
  const uint B_chunk_col = threadIdx.x % (BN/4);

  // we call this thread, and not chunk, because the C chunk (128 x 128) is bigger than the number of threads in the block
  const uint C_col_thread = threadIdx.x % (BN/TN); // 0 to 15 - each thread contributes to (8) col indices in C
  const uint C_row_thread = threadIdx.x / (BN/TN); // 0 to 15 - each thread contributes to (8) row indices in C

  float tmp[TM][TN] = {0.0f}; // TMxTN values per thread
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for(uint tile=0; tile < K; tile += BK)
  {
    float4 tmpA =
      reinterpret_cast<float4 *> (&A[(C_block_row + A_chunk_row) * K + tile + A_chunk_col * 4])[0];
    A_s[A_chunk_col * 4 + 0][A_chunk_row] = tmpA.x;
    A_s[A_chunk_col * 4 + 1][A_chunk_row] = tmpA.y;
    A_s[A_chunk_col * 4 + 2][A_chunk_row] = tmpA.z;
    A_s[A_chunk_col * 4 + 3][A_chunk_row] = tmpA.w;

    reinterpret_cast<float4 *> (&B_s[B_chunk_row][B_chunk_col * 4])[0] =
      reinterpret_cast<float4 *> (&B[(B_chunk_row + tile) * N + C_block_col + B_chunk_col * 4])[0];
    __syncthreads();

    for (uint i = 0; i < BK; ++i) {

      for (uint tm = 0; tm < TM; ++tm)
      {
        regM[tm] = A_s[i][C_row_thread * TM + tm];
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
    for (uint tn = 0; tn < TN; tn += 4)
    {
      const uint base_col = C_block_col + C_col_thread * TN + tn;

      // load C vector into registers
      float4 tmpC = reinterpret_cast<float4 *>(&C[base_row * N + base_col])[0];
      // perform GEMM update in reg
      tmpC.x = alpha * tmp[tm][tn + 0] + beta * tmpC.x;
      tmpC.y = alpha * tmp[tm][tn + 1] + beta * tmpC.y;
      tmpC.z = alpha * tmp[tm][tn + 2] + beta * tmpC.z;
      tmpC.w = alpha * tmp[tm][tn + 3] + beta * tmpC.w;
      // write back
      reinterpret_cast<float4 *>(&C[base_row * N + base_col])[0] = tmpC;
    }
  }
}
