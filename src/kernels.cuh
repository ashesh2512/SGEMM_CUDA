#pragma once

#include "kernels/10_kernel_warptiling.cuh"
#include "kernels/11_kernel_double_buffering.cuh"
#include "kernels/12_kernel_double_buffering.cuh"
#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/7_kernel_resolve_bank_conflicts.cuh"
#include "kernels/8_kernel_bank_extra_col.cuh"
#include "kernels/9_kernel_autotuned.cuh"

#include "my_kernels/101_my_sgemm_naive.cuh"
#include "my_kernels/102_my_sgemm_global_coalesced.cuh"
#include "my_kernels/103_my_sgemm_shared_mem.cuh"
#include "my_kernels/104_my_sgemm_1D_multi_elems.cuh"