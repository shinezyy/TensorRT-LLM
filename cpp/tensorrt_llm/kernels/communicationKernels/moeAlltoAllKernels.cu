/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <type_traits>
#include <vector>

// For SM90+ cluster support
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#include <cuda/barrier>
#define CLUSTER_SUPPORTED 1
#else
#define CLUSTER_SUPPORTED 0
#endif

// For cp.async support (SM80+)
#include <cuda_pipeline.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

#define ENABLE_DEBUG_PRINT 0
#define DISABLE_SYNC_FOR_PROFILING 0

// Enable timing stats collection (set to 1 to enable)
#ifndef ENABLE_A2A_TIMING_STATS
#define ENABLE_A2A_TIMING_STATS 1
#endif

#ifndef DISABLE_TIMEOUT
#define DISABLE_TIMEOUT 0
#endif

// Macros for concise launch-time specialization
#define SWITCH_BOOL(flag, NAME, ...)                                                                                   \
    if (flag)                                                                                                          \
    {                                                                                                                  \
        constexpr bool NAME = true;                                                                                    \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        constexpr bool NAME = false;                                                                                   \
        __VA_ARGS__                                                                                                    \
    }

#define SWITCH_TOP_K(top_k, TOP_K, ...)                                                                                \
    switch (top_k)                                                                                                     \
    {                                                                                                                  \
    case 22:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 22;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 16:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 16;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 10:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 10;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 8:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 8;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 6:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 6;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 4:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 4;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 2:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 2;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 1:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 1;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Unsupported top_k");                                                              \
    }                                                                                                                  \
    }

#define SWITCH_DTYPE(dtype, TYPE, ...)                                                                                 \
    switch (dtype)                                                                                                     \
    {                                                                                                                  \
    case nvinfer1::DataType::kHALF:                                                                                    \
    {                                                                                                                  \
        using TYPE = half;                                                                                             \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case nvinfer1::DataType::kBF16:                                                                                    \
    {                                                                                                                  \
        using TYPE = __nv_bfloat16;                                                                                    \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case nvinfer1::DataType::kFLOAT:                                                                                   \
    {                                                                                                                  \
        using TYPE = float;                                                                                            \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Unsupported dtype for moe_a2a_combine");                                          \
    }                                                                                                                  \
    }

#define SWITCH_POLICY(one_block_per_token, POLICY, ...)                                                                \
    if (one_block_per_token)                                                                                           \
    {                                                                                                                  \
        using POLICY = BlockPolicy;                                                                                    \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        using POLICY = WarpPolicy;                                                                                     \
        __VA_ARGS__                                                                                                    \
    }

#define SWITCH_CLUSTER_SIZE(cluster_size, CLUSTER_SIZE, ...)                                                           \
    switch (cluster_size)                                                                                              \
    {                                                                                                                  \
    case 2:                                                                                                            \
    {                                                                                                                  \
        constexpr int CLUSTER_SIZE = 2;                                                                                \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 4:                                                                                                            \
    {                                                                                                                  \
        constexpr int CLUSTER_SIZE = 4;                                                                                \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 8:                                                                                                            \
    {                                                                                                                  \
        constexpr int CLUSTER_SIZE = 8;                                                                                \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Unsupported cluster_size (must be 2, 4, or 8)");                                  \
    }                                                                                                                  \
    }

#if DISABLE_TIMEOUT
#define check_timeout(s) false
#else
// 300 * 2000 MHz - should be high enough on any GPU but will prevent a hang
#define check_timeout(s) ((clock64() - (s)) > (300ll * 2000ll * 1000ll * 1000ll))
#endif

// ============================================================================
// Helper Functions for Expert-to-Rank Mapping
// ============================================================================

__device__ int compute_target_rank_id(int expert_id, int num_experts_per_rank)
{
    // Compute which rank owns a given expert using contiguous partitioning
    // Experts are divided evenly across EP ranks:
    // - Rank 0 gets experts [0, num_experts_per_rank)
    // - Rank 1 gets experts [num_experts_per_rank, 2*num_experts_per_rank)
    // - etc.
    // Example: 32 experts, 4 ranks -> 8 experts per rank
    // - Rank 0: experts 0-7
    // - Rank 1: experts 8-15
    // - Rank 2: experts 16-23
    // - Rank 3: experts 24-31
    return expert_id / num_experts_per_rank;
}

// ============================================================================
// Helper Functions for Vectorized Memory Operations
// ============================================================================

struct WarpPolicy
{
    __device__ static int stride()
    {
        return warpSize;
    }

    __device__ static int offset()
    {
        return (threadIdx.x % warpSize);
    }

    __device__ static int token_idx()
    {
        return (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    }

    __device__ static void sync()
    {
        __syncwarp();
    }
};

struct BlockPolicy
{
    __device__ static int stride()
    {
        return blockDim.x;
    }

    __device__ static int offset()
    {
        return threadIdx.x;
    }

    __device__ static int token_idx()
    {
        return blockIdx.x;
    }

    __device__ static void sync()
    {
        __syncthreads();
    }
};

template <int VEC_SIZE, typename ThreadingPolicy>
__device__ void vectorized_copy_impl(void* dst, void const* src, int size)
{
    using flashinfer::vec_t;

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

    int const stride = ThreadingPolicy::stride() * VEC_SIZE;

    for (int offset = ThreadingPolicy::offset() * VEC_SIZE; offset < size; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> v;
        v.load(src_ptr + offset);
        v.store(dst_ptr + offset);
    }
}

template <typename ThreadingPolicy>
__device__ void vectorized_copy(void* dst, void const* src, int size)
{
    if (size % 16 == 0)
    {
        vectorized_copy_impl<16, ThreadingPolicy>(dst, src, size);
    }
    else if (size % 8 == 0)
    {
        vectorized_copy_impl<8, ThreadingPolicy>(dst, src, size);
    }
    else if (size % 4 == 0)
    {
        vectorized_copy_impl<4, ThreadingPolicy>(dst, src, size);
    }
    else if (size % 2 == 0)
    {
        vectorized_copy_impl<2, ThreadingPolicy>(dst, src, size);
    }
    else
    {
        vectorized_copy_impl<1, ThreadingPolicy>(dst, src, size);
    }
}

// Vectorized dispatch: load one vec from source and write to up to TOP_K destinations
template <int VEC_SIZE, int TOP_K, typename ThreadingPolicy>
__device__ void vectorized_dispatch_impl(uint8_t const* src_ptr, int bytes_per_token, int rank_id,
    int max_tokens_per_rank, int payload_idx, DispatchKernelPointers const& ptrs, int const* topk_target_ranks,
    int const* topk_send_indices)
{
    using flashinfer::vec_t;

    // Precompute destination base pointers per k
    uint8_t* dst_base_k[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
    {
        int dst_idx_k = topk_send_indices[k];
        int target_rank_k = topk_target_ranks[k];
        if (dst_idx_k < 0)
        {
            dst_base_k[k] = nullptr;
            continue;
        }
        uint8_t* dst_data = static_cast<uint8_t*>(ptrs.recv_buffers[target_rank_k][payload_idx]);
        size_t base_source_rank
            = static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank) + static_cast<size_t>(dst_idx_k);
        size_t base_token = base_source_rank * static_cast<size_t>(bytes_per_token);
        dst_base_k[k] = dst_data + base_token;
    }

    // TODO: process all payloads. index could be reused.
    int const stride = ThreadingPolicy::stride() * VEC_SIZE;
    for (int offset = ThreadingPolicy::offset() * VEC_SIZE; offset < bytes_per_token; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> v;
        v.load(src_ptr + offset);

#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            uint8_t* dst_base = dst_base_k[k];
            if (dst_base == nullptr)
            {
                continue;
            }
            v.store(dst_base + offset);
        }
    }
}

template <int TOP_K, typename ThreadingPolicy>
__device__ void vectorized_dispatch(uint8_t const* src_ptr, int bytes_per_token, int rank_id, int max_tokens_per_rank,
    int payload_idx, DispatchKernelPointers const& ptrs, int const* topk_target_ranks, int const* topk_send_indices)
{
    if (bytes_per_token % 16 == 0)
    {
        vectorized_dispatch_impl<16, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 8 == 0)
    {
        vectorized_dispatch_impl<8, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 4 == 0)
    {
        vectorized_dispatch_impl<4, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 2 == 0)
    {
        vectorized_dispatch_impl<2, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else
    {
        vectorized_dispatch_impl<1, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
}

__global__ void moeA2APrepareDispatchKernel(
    int* send_counters, int* local_token_counter, int ep_size, uint32_t* flag_val_ptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Zero send_counters
    if (idx < ep_size)
    {
        send_counters[idx] = 0;
    }
    // Zero local_token_counter and increment flag_val
    if (idx == 0)
    {
        *local_token_counter = 0;
        // Increment flag_val for this dispatch round
        *flag_val_ptr = *flag_val_ptr + 1;
    }
}

// ============================================================================
// Dispatch Kernels
// ============================================================================

template <typename ThreadingPolicy, int TOP_K, bool ENABLE_EPLB>
__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, TOP_K]
    const DispatchKernelPointers ptrs,                                      // Struct containing all kernel pointers
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int local_num_tokens, int rank_id, int ep_size, int num_experts, int eplb_stats_num_experts)
{
#if ENABLE_A2A_TIMING_STATS
    // Timing strategy:
    // - Block 0 records routing and data phases (via shared memory)
    // - The is_last_token block records sync phase (via global timing_stats directly)
    // - routing/data times are written to timing_stats atomically from block 0
    __shared__ uint64_t smem_timing[3]; // [t_start, t_routing_end, t_data_end]
    bool const is_block0 = (blockIdx.x == 0);
    if (is_block0 && threadIdx.x == 0)
    {
        smem_timing[0] = clock64(); // t_start
    }
#endif

    int thread_idx = ThreadingPolicy::offset();
    int local_token_idx = ThreadingPolicy::token_idx();

    if (local_num_tokens == 0)
    {
        // Special case: If local_num_tokens == 0,
        // we need to keep the threads where local_token_idx == 0 alive to participate in the synchronization.
        // Other threads should return.
        if (local_token_idx > 0)
            return;
    }
    else
    {
        // Threads that do not have a token to process should return.
        if (local_token_idx >= local_num_tokens)
            return;

        // Prepare per-policy shared-memory tiles for this token
        extern __shared__ int smem[];
        int* smem_topk_target_ranks;
        int* smem_topk_send_indices;
        int warps_per_block = blockDim.x / warpSize;
        if constexpr (std::is_same<ThreadingPolicy, WarpPolicy>::value)
        {
            int lane_id = threadIdx.x / warpSize;
            smem_topk_target_ranks = smem + lane_id * TOP_K;
            smem_topk_send_indices = smem + warps_per_block * TOP_K + lane_id * TOP_K;
        }
        else
        {
            smem_topk_target_ranks = smem;
            smem_topk_send_indices = smem + TOP_K;
        }

        uint64_t already_copied = 0;
        int num_experts_per_rank = num_experts / ep_size;
        for (int k = 0; k < TOP_K; k++)
        {
            int expert_id = token_selected_experts[local_token_idx * TOP_K + k];
            // Use contiguous partitioning to determine target rank
            int target_rank = compute_target_rank_id(expert_id, num_experts_per_rank);

            if (already_copied & (1ULL << target_rank))
            {
                if (thread_idx == 0)
                {
                    ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = -1;
                    ptrs.topk_send_indices[local_token_idx * TOP_K + k] = -1;
                    // Mirror to shared memory immediately
                    smem_topk_target_ranks[k] = -1;
                    smem_topk_send_indices[k] = -1;
                }
                continue;
            }

            // Only one thread per warp should increment the counter
            int dst_token_idx;
            if (thread_idx == 0)
            {
                dst_token_idx = atomicAdd(&ptrs.send_counters[target_rank], 1);

                ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = target_rank;
                ptrs.topk_send_indices[local_token_idx * TOP_K + k] = dst_token_idx;
                // Mirror to shared memory immediately
                smem_topk_target_ranks[k] = target_rank;
                smem_topk_send_indices[k] = dst_token_idx;
            }
            already_copied |= 1ULL << target_rank;
        }
        // Sync before dispatching data
        ThreadingPolicy::sync();

#if ENABLE_A2A_TIMING_STATS
        if (is_block0 && threadIdx.x == 0)
        {
            smem_timing[1] = clock64(); // t_routing_end
        }
#endif

        // Read staged routing once into registers per thread
        int topk_target_ranks[TOP_K];
        int topk_send_indices[TOP_K];
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            topk_target_ranks[k] = smem_topk_target_ranks[k];
            topk_send_indices[k] = smem_topk_send_indices[k];
        }

        // Perform a single source load and TOP_K fanout per payload
        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]);
            int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];
            uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;

            vectorized_dispatch<TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                payload_idx, ptrs, topk_target_ranks, topk_send_indices);
        }

        ThreadingPolicy::sync();

#if ENABLE_A2A_TIMING_STATS
        if (is_block0 && threadIdx.x == 0)
        {
            smem_timing[2] = clock64(); // t_data_end
            // Write routing and data times to timing_stats immediately
            if (ptrs.timing_stats != nullptr)
            {
                uint64_t t_start = smem_timing[0];
                uint64_t t_routing_end = smem_timing[1];
                uint64_t t_data_end = smem_timing[2];
                ptrs.timing_stats[rank_id].dispatch_routing_cycles = t_routing_end - t_start;
                ptrs.timing_stats[rank_id].dispatch_data_cycles = t_data_end - t_routing_end;
                ptrs.timing_stats[rank_id].rank_id = rank_id;
                ptrs.timing_stats[rank_id].local_num_tokens = local_num_tokens;
            }
        }
#endif
    }

    bool is_first_warp = threadIdx.x / warpSize == 0;
    if (is_first_warp)
    {
        int lane_id = threadIdx.x % warpSize;

        bool is_last_token = false;
        if (lane_id == 0)
        {
            if (local_num_tokens != 0)
            {
                int cnt = atomicAdd(ptrs.local_token_counter, 1);
                is_last_token = cnt + 1 == local_num_tokens;
            }
            else
            {
                is_last_token = true;
            }
        }
        is_last_token = __shfl_sync(0xffffffff, is_last_token, 0);

        if (is_last_token)
        {
#if ENABLE_A2A_TIMING_STATS
            uint64_t t_sync_start = clock64();
#endif

// Store send_counters to recv_counters
#pragma unroll 1 // No unroll as one iter is typically enough
            for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize)
            {
                int send_count = ptrs.send_counters[target_rank];
                ptrs.recv_counters[target_rank][rank_id] = send_count;
            }

            if constexpr (ENABLE_EPLB)
            {
                // Write local stats into peer buffers before the release fence below.
#pragma unroll 1
                for (int target_rank = 0; target_rank < ep_size; ++target_rank)
                {
                    int* target_stats = ptrs.eplb_gathered_stats[target_rank];
                    for (int expert_id = lane_id; expert_id < eplb_stats_num_experts; expert_id += warpSize)
                    {
                        int stat_val = ptrs.eplb_local_stats[expert_id];
                        target_stats[rank_id * eplb_stats_num_experts + expert_id] = stat_val;
                    }
                }
            }

#if !DISABLE_SYNC_FOR_PROFILING
            uint32_t expected_value = *ptrs.flag_val;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            // .acquire and .release qualifiers for fence instruction require sm_90 or higher.
            asm volatile("fence.release.sys;");
#else
            asm volatile("fence.acq_rel.sys;");
#endif
#pragma unroll 1 // No unroll as one iter is typically enough
            for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize)
            {
                uint32_t* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
                asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));

#if ENABLE_DEBUG_PRINT
                printf("dispatch: +++Rank %d setting completion flag to %d for rank %d\n", rank_id, expected_value,
                    target_rank);
#endif
            }

#pragma unroll 1 // No unroll
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                bool flag_set = false;
                auto s = clock64();
                do
                {
                    uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
                    uint32_t flag_value;
                    // Acquire load to ensure visibility of peer's release-store
                    asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
                    printf(
                        "combine: ---Rank %d received completion flag from rank %d, flag_value: %d, expected_value: "
                        "%d, address: %p\n",
                        rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
                    flag_set = flag_value == expected_value;
                } while (!flag_set && !check_timeout(s));

                if (__builtin_expect(!flag_set, 0))
                {
                    printf("dispatch: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id,
                        peer_rank);
                    asm volatile("trap;");
                    return;
                }
            }
#endif

#if ENABLE_A2A_TIMING_STATS
            // Record sync end time (only lane 0)
            if (lane_id == 0 && ptrs.timing_stats != nullptr)
            {
                uint64_t t_sync_end = clock64();
                ptrs.timing_stats[rank_id].dispatch_sync_cycles = t_sync_end - t_sync_start;
            }
#endif
        }
    }
}

void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params)
{
    moeA2APrepareDispatchKernel<<<1, params.ep_size, 0, params.stream>>>(
        params.send_counters, params.local_token_counter, params.ep_size, params.flag_val);
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params)
{
    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.local_num_tokens >= 0);
    TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);

    // Check if timing stats is enabled via environment variable
    bool const enableTimingStats = tensorrt_llm::common::getEnvMoeA2ATimingStats();
    MoeA2ATimingStats* deviceTimingStats = nullptr;
    if (enableTimingStats)
    {
        // Allocate device buffer for timing stats (one per rank)
        TLLM_CUDA_CHECK(cudaMallocAsync(&deviceTimingStats, sizeof(MoeA2ATimingStats) * params.ep_size, params.stream));
        TLLM_CUDA_CHECK(cudaMemsetAsync(deviceTimingStats, 0, sizeof(MoeA2ATimingStats) * params.ep_size, params.stream));
    }

    // Prepare kernel pointers struct
    DispatchKernelPointers kernel_ptrs = {};

    // Fill source data pointers and payload sizes
    for (int i = 0; i < params.num_payloads; i++)
    {
        kernel_ptrs.src_data_ptrs[i] = params.payloads[i].src_data;
        kernel_ptrs.payload_bytes_per_token[i]
            = params.payloads[i].element_size * params.payloads[i].elements_per_token;
    }

    // Fill receive buffer pointers
    for (int target_rank = 0; target_rank < params.ep_size; target_rank++)
    {
        kernel_ptrs.recv_counters[target_rank] = params.recv_counters[target_rank];
        kernel_ptrs.eplb_gathered_stats[target_rank] = params.eplb_gathered_stats[target_rank];
        for (int payload = 0; payload < params.num_payloads; payload++)
        {
            kernel_ptrs.recv_buffers[target_rank][payload] = params.recv_buffers[target_rank][payload];
        }
    }

    // Copy completion flag pointers
    for (int i = 0; i < params.ep_size; i++)
    {
        kernel_ptrs.completion_flags[i] = params.completion_flags[i];
    }
    kernel_ptrs.flag_val = params.flag_val;

    // Copy communication tracking pointers
    kernel_ptrs.send_counters = params.send_counters;
    kernel_ptrs.local_token_counter = params.local_token_counter;
    kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
    kernel_ptrs.topk_send_indices = params.topk_send_indices;
    kernel_ptrs.eplb_local_stats = params.eplb_local_stats;

    // Copy timing stats pointer (use device buffer if env var enabled, otherwise use params)
    kernel_ptrs.timing_stats = enableTimingStats ? deviceTimingStats : params.timing_stats;

    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ADispatchBlockSize();
    constexpr int kWarpSize = 32;
    int const kWarpsPerBlock = kBlockSize / kWarpSize;

    // Configure kernel launch
    if (params.one_block_per_token)
    {
        int grid_size = params.local_num_tokens;
        // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
        if (grid_size == 0)
        {
            grid_size = 1;
        }
        int shared_bytes = 2 * params.top_k * (int) sizeof(int);
        SWITCH_BOOL(params.enable_eplb, EPLB_STATS,
            SWITCH_TOP_K(params.top_k, TOP_K,
                moeA2ADispatchKernel<BlockPolicy, TOP_K, EPLB_STATS>
                <<<grid_size, kBlockSize, shared_bytes, params.stream>>>(params.token_selected_experts, kernel_ptrs,
                    params.num_payloads, params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank,
                    params.ep_size, params.num_experts, params.eplb_stats_num_experts)))
    }
    else
    {
        int grid_size = ceilDiv(params.local_num_tokens, kWarpsPerBlock);
        // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
        if (grid_size == 0)
        {
            grid_size = 1;
        }
        int shared_bytes = 2 * kWarpsPerBlock * params.top_k * (int) sizeof(int);
        SWITCH_BOOL(params.enable_eplb, EPLB_STATS,
            SWITCH_TOP_K(params.top_k, TOP_K,
                moeA2ADispatchKernel<WarpPolicy, TOP_K, EPLB_STATS>
                <<<grid_size, kBlockSize, shared_bytes, params.stream>>>(params.token_selected_experts, kernel_ptrs,
                    params.num_payloads, params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank,
                    params.ep_size, params.num_experts, params.eplb_stats_num_experts)))
    }

    // If timing stats enabled via env var, sync, copy back to host, and print
    if (enableTimingStats && deviceTimingStats != nullptr)
    {
        TLLM_CUDA_CHECK(cudaStreamSynchronize(params.stream));
        std::vector<MoeA2ATimingStats> hostTimingStats(params.ep_size);
        TLLM_CUDA_CHECK(cudaMemcpy(hostTimingStats.data(), deviceTimingStats,
            sizeof(MoeA2ATimingStats) * params.ep_size, cudaMemcpyDeviceToHost));

        // Print dispatch timing for this rank only
        float const gpuFreqGhz = tensorrt_llm::common::getEnvMoeA2AGpuFreqGhz();
        float const cyclesToUs = 1.0f / (gpuFreqGhz * 1000.0f);
        MoeA2ATimingStats const& s = hostTimingStats[params.ep_rank];
        float dispatchRouting = s.dispatch_routing_cycles * cyclesToUs;
        float dispatchData = s.dispatch_data_cycles * cyclesToUs;
        float dispatchSync = s.dispatch_sync_cycles * cyclesToUs;
        float dispatchTotal = dispatchRouting + dispatchData + dispatchSync;

        printf("[MoE A2A Dispatch Timing] Rank %d (tokens=%d): Routing=%.2f us (%.1f%%), Data=%.2f us (%.1f%%), "
               "Sync=%.2f us (%.1f%%), Total=%.2f us\n",
            params.ep_rank, params.local_num_tokens, dispatchRouting, dispatchRouting / dispatchTotal * 100,
            dispatchData, dispatchData / dispatchTotal * 100, dispatchSync, dispatchSync / dispatchTotal * 100,
            dispatchTotal);

        TLLM_CUDA_CHECK(cudaFreeAsync(deviceTimingStats, params.stream));
    }
}

// ============================================================================
// Combine kernels
// ============================================================================

// Accumulate across all valid ranks into registers, then store once per segment
template <int VEC_SIZE, int TOP_K, typename ThreadingPolicy, typename T>
__device__ void vectorized_combine_impl(
    T* dst_typed_base, int size_per_token, int rank_id, int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    constexpr int elems_per_vec = VEC_SIZE / sizeof(T);
    using flashinfer::vec_t;

    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst_typed_base);

    int const stride = ThreadingPolicy::stride() * VEC_SIZE;
    int const local_token_idx = ThreadingPolicy::token_idx();

    for (int offset = ThreadingPolicy::offset() * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> acc[TOP_K];

// Unrolled K accumulation using compact top-k lists
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
            int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
            if (dst_idx < 0)
            {
                acc[k].fill(0);
                continue;
            }

            uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
            size_t base_source_rank = static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank)
                + static_cast<size_t>(dst_idx);
            size_t base_token = base_source_rank * static_cast<size_t>(size_per_token);

            // Load directly into the per-k accumulator; reduce across k below
            acc[k].load(recv_buffer + base_token + offset);
        }
        // Reduce acc[TOP_K] into acc[0]
        if constexpr (TOP_K == 22)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
            T* a8 = reinterpret_cast<T*>(&acc[8]);
            T* a9 = reinterpret_cast<T*>(&acc[9]);
            T* a10 = reinterpret_cast<T*>(&acc[10]);
            T* a11 = reinterpret_cast<T*>(&acc[11]);
            T* a12 = reinterpret_cast<T*>(&acc[12]);
            T* a13 = reinterpret_cast<T*>(&acc[13]);
            T* a14 = reinterpret_cast<T*>(&acc[14]);
            T* a15 = reinterpret_cast<T*>(&acc[15]);
            T* a16 = reinterpret_cast<T*>(&acc[16]);
            T* a17 = reinterpret_cast<T*>(&acc[17]);
            T* a18 = reinterpret_cast<T*>(&acc[18]);
            T* a19 = reinterpret_cast<T*>(&acc[19]);
            T* a20 = reinterpret_cast<T*>(&acc[20]);
            T* a21 = reinterpret_cast<T*>(&acc[21]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
                a8[j] += a9[j];
                a10[j] += a11[j];
                a12[j] += a13[j];
                a14[j] += a15[j];
                a16[j] += a17[j];
                a18[j] += a19[j];
                a20[j] += a21[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
                a8[j] += a10[j];
                a12[j] += a14[j];
                a16[j] += a18[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
                a8[j] += a12[j];
                a16[j] += a20[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a8[j];
                a0[j] += a16[j];
            }
        }
        else if constexpr (TOP_K == 16)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
            T* a8 = reinterpret_cast<T*>(&acc[8]);
            T* a9 = reinterpret_cast<T*>(&acc[9]);
            T* a10 = reinterpret_cast<T*>(&acc[10]);
            T* a11 = reinterpret_cast<T*>(&acc[11]);
            T* a12 = reinterpret_cast<T*>(&acc[12]);
            T* a13 = reinterpret_cast<T*>(&acc[13]);
            T* a14 = reinterpret_cast<T*>(&acc[14]);
            T* a15 = reinterpret_cast<T*>(&acc[15]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
                a8[j] += a9[j];
                a10[j] += a11[j];
                a12[j] += a13[j];
                a14[j] += a15[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
                a8[j] += a10[j];
                a12[j] += a14[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
                a8[j] += a12[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a8[j];
            }
        }
        else if constexpr (TOP_K == 10)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
            T* a8 = reinterpret_cast<T*>(&acc[8]);
            T* a9 = reinterpret_cast<T*>(&acc[9]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
                a8[j] += a9[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
                a0[j] += a8[j];
            }
        }
        else if constexpr (TOP_K == 8)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
            }
        }
        else if constexpr (TOP_K == 6)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a0[j] += a4[j];
            }
        }
        else if constexpr (TOP_K == 4)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
            }
        }
        else if constexpr (TOP_K == 2)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
            }
        }
        else if constexpr (TOP_K == 1)
        {
            // nothing to do
        }
        else
        {
            // Generic fallback: accumulate all into acc[0]
            T* a0 = reinterpret_cast<T*>(&acc[0]);
#pragma unroll
            for (int k = 1; k < TOP_K; ++k)
            {
                T* ak = reinterpret_cast<T*>(&acc[k]);
#pragma unroll
                for (int j = 0; j < elems_per_vec; ++j)
                {
                    a0[j] += ak[j];
                }
            }
        }

        acc[0].store(dst_bytes + offset);
    }
}

// Wrapper that selects vector width based on size_per_token alignment
template <int TOP_K, typename ThreadingPolicy, typename T>
__device__ void vectorized_combine(
    T* dst_typed_base, int size_per_token, int rank_id, int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    if (size_per_token % 16 == 0)
    {
        vectorized_combine_impl<16, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 8 == 0)
    {
        vectorized_combine_impl<8, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 4 == 0)
    {
        vectorized_combine_impl<4, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 2 == 0)
    {
        vectorized_combine_impl<2, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else
    {
        vectorized_combine_impl<1, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
}

// Copy payload to recv buffer using vectorized copy; supports warp/block token mapping
template <typename ThreadingPolicy>
__global__ void moeA2APrepareCombineKernel(uint8_t* recv_buffer_bytes, uint8_t const* payload_bytes,
    int bytes_per_token, int ep_size, int max_tokens_per_rank, uint32_t* flag_val_ptr, int const* recv_counters)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // Increment flag_val for this combine round
        *flag_val_ptr = *flag_val_ptr + 1;
    }

    if (payload_bytes == nullptr)
        return;

    int global_token_idx = ThreadingPolicy::token_idx();

    int global_token_num = ep_size * max_tokens_per_rank;
    if (global_token_idx >= global_token_num)
        return;

    // Map global_token_idx to (rank_idx, local_token_idx)
    int rank_idx = global_token_idx / max_tokens_per_rank;
    int local_token_idx = global_token_idx % max_tokens_per_rank;

    // Skip invalid tokens beyond per-rank recv count
    if (local_token_idx >= recv_counters[rank_idx])
        return;

    // Calculate source and destination pointers for this token
    size_t offset = static_cast<size_t>(global_token_idx) * bytes_per_token;
    uint8_t* dst_ptr = recv_buffer_bytes + offset;
    uint8_t const* src_ptr = payload_bytes + offset;

    // Copy one token's data using vectorized copy with policy
    vectorized_copy<ThreadingPolicy>(dst_ptr, src_ptr, bytes_per_token);
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, typename ThreadingPolicy, int TOP_K>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs, // Combine-specific struct, src_data_ptrs[0] is output
    int max_tokens_per_rank, int elements_per_token, int local_num_tokens, int rank_id, int ep_size)
{
#if ENABLE_A2A_TIMING_STATS
    // Timing variables - only first token of each rank records timing
    uint64_t t_start = 0, t_sync_end = 0, t_data_end = 0;
    bool const is_timing_thread = (ThreadingPolicy::token_idx() == 0 && ThreadingPolicy::offset() == 0);
    if (is_timing_thread)
    {
        t_start = clock64();
    }
#endif

    int local_token_idx = ThreadingPolicy::token_idx();
    int const size_per_token = elements_per_token * sizeof(T);

    if (local_num_tokens == 0)
    {
        // Special case: If local_num_tokens == 0,
        // we need to keep the threads where local_token_idx == 0 alive to participate in the synchronization.
        // Other threads should return.
        if (local_token_idx > 0)
            return;
    }
    else
    {
        // Threads that do not have a token to process should return.
        if (local_token_idx >= local_num_tokens)
            return;
    }

#if !DISABLE_SYNC_FOR_PROFILING
    // In-kernel readiness synchronization at start of combine:
    // - One warp signals readiness to all peers with current flag_val.
    // - The first warp of each block waits for all peers' readiness (equality), then __syncthreads.
    bool is_first_warp = threadIdx.x / warpSize == 0;
    if (is_first_warp)
    {
        int lane_id = threadIdx.x % warpSize;
        uint32_t expected_value = *ptrs.flag_val;

        if (blockIdx.x == 0)
        {
#pragma unroll 1 // No unroll
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                uint32_t* flag_addr = &ptrs.completion_flags[peer_rank][rank_id];
                asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));
#if ENABLE_DEBUG_PRINT
                printf("combine: +++Rank %d setting completion flag to %d for rank %d\n", rank_id, expected_value,
                    peer_rank);
#endif // ENABLE_DEBUG_PRINT
            }
        }

#pragma unroll 1 // No unroll
        for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
        {
            bool flag_set = false;
            auto s = clock64();
            do
            {
                uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
                uint32_t flag_value;
                // Acquire load to ensure visibility of peer's release-store
                asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
                printf(
                    "combine: ---Rank %d received completion flag from rank %d, flag_value: %d, expected_value: "
                    "%d, "
                    "address: %p\n",
                    rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif // ENABLE_DEBUG_PRINT
                flag_set = flag_value == expected_value;
            } while (!flag_set && !check_timeout(s));

            if (__builtin_expect(!flag_set, 0))
            {
                printf("combine: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id, peer_rank);
                asm volatile("trap;");
                return;
            }
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        // .acquire and .release qualifiers for fence instruction require sm_90 or higher.
        asm volatile("fence.acquire.sys;");
#else
        asm volatile("fence.acq_rel.sys;");
#endif
    }
    __syncthreads();
#endif // DISABLE_SYNC_FOR_PROFILING

#if ENABLE_A2A_TIMING_STATS
    if (is_timing_thread)
    {
        t_sync_end = clock64();
    }
#endif

    if (local_num_tokens == 0)
        return;

    // Get output location for this token (using src_data_ptrs[0] as output)
    T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;

    // Accumulate across ranks in registers, then store once per segment
    vectorized_combine<TOP_K, ThreadingPolicy, T>(token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs);

#if ENABLE_A2A_TIMING_STATS
    // Record data end time and write stats (only timing thread)
    if (is_timing_thread && ptrs.timing_stats != nullptr)
    {
        t_data_end = clock64();
        ptrs.timing_stats[rank_id].combine_sync_cycles = t_sync_end - t_start;
        ptrs.timing_stats[rank_id].combine_data_cycles = t_data_end - t_sync_end;
    }
#endif
}

// ============================================================================
// Cluster-based Combine Kernel (SM90+ with DSMEM)
// Uses Thread Block Clusters to parallelize remote reads across CTAs
// Each CTA in the cluster fetches a subset of TOP_K ranks, then results are
// aggregated via distributed shared memory (DSMEM)
//
// OPTIMIZATION: One-shot bulk read into shared memory
// - Allocate full size_per_token in smem (14KB << 228KB available)
// - Issue ALL remote reads first (let memory controller pipeline)
// - Single cluster.sync() after all reads complete
// - DSMEM gather and output
// - Single cluster.sync() at end
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

// Cluster timing stats (accumulated in registers, written once at end)
struct ClusterTimingAccumulator
{
    uint64_t remote_read_cycles = 0;
    uint64_t sync1_cycles = 0;
    uint64_t dsmem_cycles = 0;
    uint64_t sync2_cycles = 0;
    int iterations = 0;
};

// ============================================================================
// Optimized One-shot Implementation
// Key insight: Load ALL data into shared memory in one pass, then single sync
// ============================================================================

template <int VEC_SIZE, int TOP_K, int CLUSTER_SIZE, typename T, bool COLLECT_TIMING>
__device__ void vectorized_combine_cluster_impl(T* dst_typed_base, int size_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs, uint8_t* smem_partial,
    ClusterTimingAccumulator* timing_acc)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    constexpr int elems_per_vec = VEC_SIZE / sizeof(T);
    using flashinfer::vec_t;

    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst_typed_base);

    int const stride = blockDim.x * VEC_SIZE;
    int const local_token_idx = blockIdx.x / CLUSTER_SIZE;
    int const cluster_rank = cluster.block_rank();

    constexpr int RANKS_PER_CTA = (TOP_K + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    int const k_start = cluster_rank * RANKS_PER_CTA;
    int const k_end = min(k_start + RANKS_PER_CTA, TOP_K);

    bool const is_timing_thread = COLLECT_TIMING && (cluster_rank == 0) && (threadIdx.x == 0);

    uint64_t t0, t1, t2, t3;

    // =========================================================================
    // PHASE 1: Bulk read ALL data into shared memory (one-shot)
    // All threads cooperatively load the entire size_per_token into smem
    // Memory controller will pipeline the NVLink requests
    // =========================================================================
    if (is_timing_thread)
        t0 = clock64();

    // Process all offsets - load from remote and accumulate into smem
    for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> acc;
        acc.fill(0);

        // Load from all assigned ranks and accumulate in registers
        // These loads are issued back-to-back, memory controller will pipeline
        #pragma unroll
        for (int k = k_start; k < k_end && k < TOP_K; ++k)
        {
            int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
            int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];

            if (dst_idx >= 0)
            {
                uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
                size_t base = static_cast<size_t>(rank_id) * max_tokens_per_rank + dst_idx;
                base *= size_per_token;

                vec_t<uint8_t, VEC_SIZE> v;
                v.load(recv_buffer + base + offset);

                // Accumulate
                T* a = reinterpret_cast<T*>(&acc);
                T* val = reinterpret_cast<T*>(&v);
                #pragma unroll
                for (int j = 0; j < elems_per_vec; ++j)
                {
                    a[j] += val[j];
                }
            }
        }

        // Store accumulated result to shared memory
        acc.store(smem_partial + offset);
    }

    if (is_timing_thread)
        t1 = clock64();

    // =========================================================================
    // PHASE 2: Single cluster-wide sync (all CTAs have their data in smem)
    // =========================================================================
    cluster.sync();

    if (is_timing_thread)
        t2 = clock64();

    // =========================================================================
    // PHASE 3: CTA 0 gathers all partial sums via DSMEM and writes output
    // =========================================================================
    if (cluster_rank == 0)
    {
        for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
        {
            // Load own partial sum
            vec_t<uint8_t, VEC_SIZE> final_acc;
            final_acc.load(smem_partial + offset);

            // Gather from other CTAs via distributed shared memory
            #pragma unroll
            for (int c = 1; c < CLUSTER_SIZE; ++c)
            {
                uint8_t* other_smem = cluster.map_shared_rank(smem_partial, c);

                vec_t<uint8_t, VEC_SIZE> other;
                other.load(other_smem + offset);

                T* f = reinterpret_cast<T*>(&final_acc);
                T* o = reinterpret_cast<T*>(&other);
                #pragma unroll
                for (int j = 0; j < elems_per_vec; ++j)
                {
                    f[j] += o[j];
                }
            }

            // Store final result to global memory
            final_acc.store(dst_bytes + offset);
        }
    }

    if (is_timing_thread)
        t3 = clock64();

    // =========================================================================
    // PHASE 4: Final sync (ensure CTA 0 has finished writing before next iter)
    // =========================================================================
    cluster.sync();

    if (is_timing_thread)
    {
        uint64_t t4 = clock64();
        timing_acc->remote_read_cycles = t1 - t0;
        timing_acc->sync1_cycles = t2 - t1;
        timing_acc->dsmem_cycles = t3 - t2;
        timing_acc->sync2_cycles = t4 - t3;
        timing_acc->iterations = 1;  // One-shot, single iteration
    }
}

// ============================================================================
// 方案C: Break Dependency Chain + Tree Reduce Implementation
// Key insight: Load all data into register buffers FIRST, then accumulate
// Tree reduce with polling replaces expensive cluster.sync():
//   CLUSTER_SIZE==8: 3 rounds (0+1,2+3,4+5,6+7) -> ((0+1)+(2+3),(4+5)+(6+7)) -> final
//   CLUSTER_SIZE==4: 2 rounds (0+1,2+3) -> ((0+1)+(2+3))
//   CLUSTER_SIZE<=2: Falls back to cluster.sync() + DSMEM gather
// ============================================================================

template <int VEC_SIZE, int TOP_K, int CLUSTER_SIZE, typename T, bool COLLECT_TIMING>
__device__ void vectorized_combine_cluster_break_dep_impl(T* dst_typed_base, int size_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs, uint8_t* smem_partial,
    ClusterTimingAccumulator* timing_acc)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    constexpr int elems_per_vec = VEC_SIZE / sizeof(T);
    using flashinfer::vec_t;

    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst_typed_base);

    int const stride = blockDim.x * VEC_SIZE;
    int const local_token_idx = blockIdx.x / CLUSTER_SIZE;
    int const cluster_rank = cluster.block_rank();

    constexpr int RANKS_PER_CTA = (TOP_K + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    int const k_start = cluster_rank * RANKS_PER_CTA;
    int const k_end = min(k_start + RANKS_PER_CTA, TOP_K);

    bool const is_timing_thread = COLLECT_TIMING && (cluster_rank == 0) && (threadIdx.x == 0);

    uint64_t t0, t1, t2, t3;

    // Shared memory flag for tree reduce synchronization (used when CLUSTER_SIZE >= 4)
    __shared__ int reduce_flag;

    // Pre-load routing information into registers (constant for all offsets)
    int target_ranks[RANKS_PER_CTA];
    int dst_indices[RANKS_PER_CTA];
    uint8_t const* recv_buffer_bases[RANKS_PER_CTA];

    #pragma unroll
    for (int i = 0; i < RANKS_PER_CTA; ++i)
    {
        int k = k_start + i;
        if (k < k_end && k < TOP_K)
        {
            target_ranks[i] = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
            dst_indices[i] = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
            if (dst_indices[i] >= 0)
            {
                recv_buffer_bases[i] = static_cast<uint8_t const*>(ptrs.recv_buffers[target_ranks[i]][0]);
                size_t base = static_cast<size_t>(rank_id) * max_tokens_per_rank + dst_indices[i];
                recv_buffer_bases[i] += base * size_per_token;
            }
            else
            {
                recv_buffer_bases[i] = nullptr;
            }
        }
        else
        {
            target_ranks[i] = -1;
            dst_indices[i] = -1;
            recv_buffer_bases[i] = nullptr;
        }
    }

    // Initialize flag (only needed for tree reduce path: CLUSTER_SIZE >= 4)
    if constexpr (CLUSTER_SIZE >= 4)
    {
        if (threadIdx.x == 0)
        {
            reduce_flag = 0;
        }
        __syncthreads();
    }

    // =========================================================================
    // PHASE 1: Bulk read ALL data into shared memory with dependency-free loads
    // Issue all loads FIRST into a register buffer, THEN accumulate
    // =========================================================================
    if (is_timing_thread)
        t0 = clock64();

    for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        // Register buffer for all loads (break dependency chain!)
        vec_t<uint8_t, VEC_SIZE> bufs[RANKS_PER_CTA];

        // PHASE 1a: Issue ALL loads without any accumulation (no dependency)
        #pragma unroll
        for (int i = 0; i < RANKS_PER_CTA; ++i)
        {
            if (recv_buffer_bases[i] != nullptr)
            {
                bufs[i].load(recv_buffer_bases[i] + offset);
            }
            else
            {
                bufs[i].fill(0);
            }
            // NOTE: No accumulation here - this breaks the dependency chain!
        }

        // PHASE 1b: Accumulate all buffers (all dependencies are local registers now)
        vec_t<uint8_t, VEC_SIZE> acc;
        acc.fill(0);

        #pragma unroll
        for (int i = 0; i < RANKS_PER_CTA; ++i)
        {
            T* a = reinterpret_cast<T*>(&acc);
            T* b = reinterpret_cast<T*>(&bufs[i]);
            #pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a[j] += b[j];
            }
        }

        // Store accumulated result to shared memory
        acc.store(smem_partial + offset);
    }

    // Ensure all threads in this CTA have finished writing to shared memory
    __syncthreads();

    if (is_timing_thread)
        t1 = clock64();

    // =========================================================================
    // PHASE 2: Reduction - Tree Reduce (4/8-CTA) or cluster.sync + gather (<=2)
    // =========================================================================

    if constexpr (CLUSTER_SIZE == 8)
    {
        // =====================================================================
        // Tree Reduce for 8-CTA cluster (with handshake protocol)
        // Round 1: 0+1, 2+3, 4+5, 6+7
        // Round 2: (0+1)+(2+3), (4+5)+(6+7)
        // Round 3: Final sum at CTA 0
        // Handshake: signaling CTA sets flag=1, waits for flag=0 (ack from poller)
        //            polling CTA polls flag=1, does work, resets remote flag=0
        // =====================================================================

        // ---------------------------------------------------------------------
        // Round 1: CTAs 1,3,5,7 signal; CTAs 0,2,4,6 poll and accumulate
        // ---------------------------------------------------------------------
        {
            int neighbor = cluster_rank ^ 1;  // 0<->1, 2<->3, 4<->5, 6<->7
            int* remote_flag = cluster.map_shared_rank(&reduce_flag, neighbor);

            if (cluster_rank % 2 == 1)
            {
                // Odd CTAs: signal completion, then wait for ack
                if (threadIdx.x == 0)
                {
                    reduce_flag = 1;
                    __threadfence_block();
                    // Wait for polling CTA to reset our flag (acknowledgment)
                    while (atomicAdd(&reduce_flag, 0) != 0) {}
                }
                __syncthreads();
            }
            else
            {
                // Even CTAs: poll neighbor, accumulate, then send ack
                if (threadIdx.x == 0)
                {
                    while (atomicAdd(remote_flag, 0) == 0) {}
                }
                __syncthreads();

                // Accumulate neighbor's data via DSMEM
                uint8_t* neighbor_smem = cluster.map_shared_rank(smem_partial, neighbor);
                for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
                {
                    vec_t<uint8_t, VEC_SIZE> local_acc;
                    local_acc.load(smem_partial + offset);

                    vec_t<uint8_t, VEC_SIZE> neighbor_data;
                    neighbor_data.load(neighbor_smem + offset);

                    T* l = reinterpret_cast<T*>(&local_acc);
                    T* n = reinterpret_cast<T*>(&neighbor_data);
                    #pragma unroll
                    for (int j = 0; j < elems_per_vec; ++j)
                    {
                        l[j] += n[j];
                    }

                    local_acc.store(smem_partial + offset);
                }
                __syncthreads();

                // Send ack by resetting remote flag
                if (threadIdx.x == 0)
                {
                    atomicExch(remote_flag, 0);
                }
            }
        }

        // ---------------------------------------------------------------------
        // Round 2: CTAs 2,6 signal; CTAs 0,4 poll and accumulate
        // ---------------------------------------------------------------------
        if (cluster_rank == 2 || cluster_rank == 6 || cluster_rank == 0 || cluster_rank == 4)
        {
            int neighbor = cluster_rank ^ 2;  // 0<->2, 4<->6
            int* remote_flag = cluster.map_shared_rank(&reduce_flag, neighbor);

            if (cluster_rank == 2 || cluster_rank == 6)
            {
                // Signal completion, then wait for ack
                if (threadIdx.x == 0)
                {
                    reduce_flag = 1;
                    __threadfence_block();
                    while (atomicAdd(&reduce_flag, 0) != 0) {}
                }
                __syncthreads();
            }
            else  // cluster_rank == 0 || cluster_rank == 4
            {
                // Poll neighbor, accumulate, then send ack
                if (threadIdx.x == 0)
                {
                    while (atomicAdd(remote_flag, 0) == 0) {}
                }
                __syncthreads();

                uint8_t* neighbor_smem = cluster.map_shared_rank(smem_partial, neighbor);
                for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
                {
                    vec_t<uint8_t, VEC_SIZE> local_acc;
                    local_acc.load(smem_partial + offset);

                    vec_t<uint8_t, VEC_SIZE> neighbor_data;
                    neighbor_data.load(neighbor_smem + offset);

                    T* l = reinterpret_cast<T*>(&local_acc);
                    T* n = reinterpret_cast<T*>(&neighbor_data);
                    #pragma unroll
                    for (int j = 0; j < elems_per_vec; ++j)
                    {
                        l[j] += n[j];
                    }

                    local_acc.store(smem_partial + offset);
                }
                __syncthreads();

                // Send ack
                if (threadIdx.x == 0)
                {
                    atomicExch(remote_flag, 0);
                }
            }
        }

        if (is_timing_thread)
            t2 = clock64();

        // ---------------------------------------------------------------------
        // Round 3: CTA 4 signals; CTA 0 polls, accumulates, and writes output
        // ---------------------------------------------------------------------
        if (cluster_rank == 4 || cluster_rank == 0)
        {
            int* remote_flag = cluster.map_shared_rank(&reduce_flag, 4);

            if (cluster_rank == 4)
            {
                // Signal completion (no need to wait for ack - last round)
                if (threadIdx.x == 0)
                {
                    reduce_flag = 1;
                    __threadfence_block();
                }
            }
            else  // cluster_rank == 0
            {
                // Poll CTA 4 and do final accumulation + output
                if (threadIdx.x == 0)
                {
                    while (atomicAdd(remote_flag, 0) == 0) {}
                }
                __syncthreads();

                uint8_t* neighbor_smem = cluster.map_shared_rank(smem_partial, 4);
                for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
                {
                    vec_t<uint8_t, VEC_SIZE> local_acc;
                    local_acc.load(smem_partial + offset);

                    vec_t<uint8_t, VEC_SIZE> neighbor_data;
                    neighbor_data.load(neighbor_smem + offset);

                    T* l = reinterpret_cast<T*>(&local_acc);
                    T* n = reinterpret_cast<T*>(&neighbor_data);
                    #pragma unroll
                    for (int j = 0; j < elems_per_vec; ++j)
                    {
                        l[j] += n[j];
                    }

                    // Write final result to global memory
                    local_acc.store(dst_bytes + offset);
                }
            }
        }

        if (is_timing_thread)
            t3 = clock64();
    }
    else if constexpr (CLUSTER_SIZE == 4)
    {
        // =====================================================================
        // Tree Reduce for 4-CTA cluster (with handshake protocol)
        // Round 1: 0+1, 2+3
        // Round 2: (0+1)+(2+3) -> CTA 0 writes output
        // =====================================================================

        // ---------------------------------------------------------------------
        // Round 1: CTAs 1,3 signal; CTAs 0,2 poll and accumulate
        // ---------------------------------------------------------------------
        {
            int neighbor = cluster_rank ^ 1;  // 0<->1, 2<->3
            int* remote_flag = cluster.map_shared_rank(&reduce_flag, neighbor);

            if (cluster_rank % 2 == 1)
            {
                // Odd CTAs: signal completion, then wait for ack
                if (threadIdx.x == 0)
                {
                    reduce_flag = 1;
                    __threadfence_block();
                    while (atomicAdd(&reduce_flag, 0) != 0) {}
                }
                __syncthreads();
            }
            else
            {
                // Even CTAs: poll neighbor, accumulate, then send ack
                if (threadIdx.x == 0)
                {
                    while (atomicAdd(remote_flag, 0) == 0) {}
                }
                __syncthreads();

                uint8_t* neighbor_smem = cluster.map_shared_rank(smem_partial, neighbor);
                for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
                {
                    vec_t<uint8_t, VEC_SIZE> local_acc;
                    local_acc.load(smem_partial + offset);

                    vec_t<uint8_t, VEC_SIZE> neighbor_data;
                    neighbor_data.load(neighbor_smem + offset);

                    T* l = reinterpret_cast<T*>(&local_acc);
                    T* n = reinterpret_cast<T*>(&neighbor_data);
                    #pragma unroll
                    for (int j = 0; j < elems_per_vec; ++j)
                    {
                        l[j] += n[j];
                    }

                    local_acc.store(smem_partial + offset);
                }
                __syncthreads();

                // Send ack
                if (threadIdx.x == 0)
                {
                    atomicExch(remote_flag, 0);
                }
            }
        }

        if (is_timing_thread)
            t2 = clock64();

        // ---------------------------------------------------------------------
        // Round 2: CTA 2 signals; CTA 0 polls, accumulates, and writes output
        // ---------------------------------------------------------------------
        if (cluster_rank == 2 || cluster_rank == 0)
        {
            int* remote_flag = cluster.map_shared_rank(&reduce_flag, 2);

            if (cluster_rank == 2)
            {
                // Signal completion (no need to wait for ack - last round)
                if (threadIdx.x == 0)
                {
                    reduce_flag = 1;
                    __threadfence_block();
                }
            }
            else  // cluster_rank == 0
            {
                // Poll CTA 2 and do final accumulation + output
                if (threadIdx.x == 0)
                {
                    while (atomicAdd(remote_flag, 0) == 0) {}
                }
                __syncthreads();

                uint8_t* neighbor_smem = cluster.map_shared_rank(smem_partial, 2);
                for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
                {
                    vec_t<uint8_t, VEC_SIZE> local_acc;
                    local_acc.load(smem_partial + offset);

                    vec_t<uint8_t, VEC_SIZE> neighbor_data;
                    neighbor_data.load(neighbor_smem + offset);

                    T* l = reinterpret_cast<T*>(&local_acc);
                    T* n = reinterpret_cast<T*>(&neighbor_data);
                    #pragma unroll
                    for (int j = 0; j < elems_per_vec; ++j)
                    {
                        l[j] += n[j];
                    }

                    // Write final result to global memory
                    local_acc.store(dst_bytes + offset);
                }
            }
        }

        if (is_timing_thread)
            t3 = clock64();
    }
    else
    {
        // =====================================================================
        // Fallback: cluster.sync() + DSMEM gather (for CLUSTER_SIZE <= 2)
        // =====================================================================
        cluster.sync();

        if (is_timing_thread)
            t2 = clock64();

        // CTA 0 gathers all partial sums via DSMEM and writes output
        if (cluster_rank == 0)
        {
            for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
            {
                // Load own partial sum
                vec_t<uint8_t, VEC_SIZE> final_acc;
                final_acc.load(smem_partial + offset);

                // Gather from other CTAs via distributed shared memory
                #pragma unroll
                for (int c = 1; c < CLUSTER_SIZE; ++c)
                {
                    uint8_t* other_smem = cluster.map_shared_rank(smem_partial, c);

                    vec_t<uint8_t, VEC_SIZE> other;
                    other.load(other_smem + offset);

                    T* f = reinterpret_cast<T*>(&final_acc);
                    T* o = reinterpret_cast<T*>(&other);
                    #pragma unroll
                    for (int j = 0; j < elems_per_vec; ++j)
                    {
                        f[j] += o[j];
                    }
                }

                // Store final result to global memory
                final_acc.store(dst_bytes + offset);
            }
        }

        if (is_timing_thread)
            t3 = clock64();
    }

    // =========================================================================
    // PHASE 3: Final sync (ensure CTA 0 has finished writing before next iter)
    // =========================================================================
    cluster.sync();

    if (is_timing_thread)
    {
        uint64_t t4 = clock64();
        timing_acc->remote_read_cycles = t1 - t0;
        timing_acc->sync1_cycles = t2 - t1;   // Tree reduce (or cluster.sync for fallback)
        timing_acc->dsmem_cycles = t3 - t2;   // Final reduce round + output (or DSMEM gather)
        timing_acc->sync2_cycles = t4 - t3;   // Final cluster.sync()
        timing_acc->iterations = 1;
    }
}

// ============================================================================
// 方案D: Async Copy (cp.async) Implementation
// Key insight: Use hardware async copy to pipeline NVLink reads
// cp.async copies data directly from global to shared memory asynchronously
// ============================================================================

template <int VEC_SIZE, int TOP_K, int CLUSTER_SIZE, typename T, bool COLLECT_TIMING>
__device__ void vectorized_combine_cluster_cpasync_impl(T* dst_typed_base, int size_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs, uint8_t* smem_partial, uint8_t* smem_async_buffers,
    ClusterTimingAccumulator* timing_acc)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    constexpr int elems_per_vec = VEC_SIZE / sizeof(T);
    using flashinfer::vec_t;

    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst_typed_base);

    int const stride = blockDim.x * VEC_SIZE;
    int const local_token_idx = blockIdx.x / CLUSTER_SIZE;
    int const cluster_rank = cluster.block_rank();

    constexpr int RANKS_PER_CTA = (TOP_K + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    int const k_start = cluster_rank * RANKS_PER_CTA;
    int const k_end = min(k_start + RANKS_PER_CTA, TOP_K);

    bool const is_timing_thread = COLLECT_TIMING && (cluster_rank == 0) && (threadIdx.x == 0);

    uint64_t t0, t1, t2, t3;

    // Pre-compute source addresses
    uint8_t const* src_addrs[RANKS_PER_CTA];
    bool valid_ranks[RANKS_PER_CTA];

    #pragma unroll
    for (int i = 0; i < RANKS_PER_CTA; ++i)
    {
        int k = k_start + i;
        if (k < k_end && k < TOP_K)
        {
            int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
            int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
            if (dst_idx >= 0)
            {
                uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
                size_t base = static_cast<size_t>(rank_id) * max_tokens_per_rank + dst_idx;
                src_addrs[i] = recv_buffer + base * size_per_token;
                valid_ranks[i] = true;
            }
            else
            {
                src_addrs[i] = nullptr;
                valid_ranks[i] = false;
            }
        }
        else
        {
            src_addrs[i] = nullptr;
            valid_ranks[i] = false;
        }
    }

    // Calculate buffer stride for async copies
    // Each rank's data goes to a separate region in shared memory
    // smem_async_buffers layout: [rank0_data][rank1_data]...[rank(RANKS_PER_CTA-1)_data]
    // Each region is size_per_token bytes

    // =========================================================================
    // PHASE 1: Issue async copies for ALL data
    // cp.async allows hardware to pipeline NVLink requests
    // =========================================================================
    if (is_timing_thread)
        t0 = clock64();

    // Issue async copies for each thread's portion of data
    for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        #pragma unroll
        for (int i = 0; i < RANKS_PER_CTA; ++i)
        {
            uint8_t* dst_smem = smem_async_buffers + i * size_per_token + offset;
            if (valid_ranks[i])
            {
                // Use cp.async for hardware-accelerated async copy
                // cp.async copies VEC_SIZE bytes from global to shared memory
                __pipeline_memcpy_async(dst_smem, src_addrs[i] + offset, VEC_SIZE);
            }
            else
            {
                // For invalid ranks, zero the buffer
                vec_t<uint8_t, VEC_SIZE> zero;
                zero.fill(0);
                zero.store(dst_smem);
            }
        }
    }

    // Commit all async copies
    __pipeline_commit();

    // Wait for all async copies to complete
    __pipeline_wait_prior(0);

    // Memory fence to ensure all threads see the data
    __syncthreads();

    // Now accumulate all data from async buffers into smem_partial
    for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> acc;
        acc.fill(0);

        #pragma unroll
        for (int i = 0; i < RANKS_PER_CTA; ++i)
        {
            uint8_t* src_smem = smem_async_buffers + i * size_per_token + offset;
            vec_t<uint8_t, VEC_SIZE> v;
            v.load(src_smem);

            T* a = reinterpret_cast<T*>(&acc);
            T* val = reinterpret_cast<T*>(&v);
            #pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a[j] += val[j];
            }
        }

        // Store accumulated result to smem_partial
        acc.store(smem_partial + offset);
    }

    if (is_timing_thread)
        t1 = clock64();

    // =========================================================================
    // PHASE 2: Single cluster-wide sync (all CTAs have their data in smem)
    // =========================================================================
    cluster.sync();

    if (is_timing_thread)
        t2 = clock64();

    // =========================================================================
    // PHASE 3: CTA 0 gathers all partial sums via DSMEM and writes output
    // =========================================================================
    if (cluster_rank == 0)
    {
        for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
        {
            // Load own partial sum
            vec_t<uint8_t, VEC_SIZE> final_acc;
            final_acc.load(smem_partial + offset);

            // Gather from other CTAs via distributed shared memory
            #pragma unroll
            for (int c = 1; c < CLUSTER_SIZE; ++c)
            {
                uint8_t* other_smem = cluster.map_shared_rank(smem_partial, c);

                vec_t<uint8_t, VEC_SIZE> other;
                other.load(other_smem + offset);

                T* f = reinterpret_cast<T*>(&final_acc);
                T* o = reinterpret_cast<T*>(&other);
                #pragma unroll
                for (int j = 0; j < elems_per_vec; ++j)
                {
                    f[j] += o[j];
                }
            }

            // Store final result to global memory
            final_acc.store(dst_bytes + offset);
        }
    }

    if (is_timing_thread)
        t3 = clock64();

    // =========================================================================
    // PHASE 4: Final sync (ensure CTA 0 has finished writing before next iter)
    // =========================================================================
    cluster.sync();

    if (is_timing_thread)
    {
        uint64_t t4 = clock64();
        timing_acc->remote_read_cycles = t1 - t0;
        timing_acc->sync1_cycles = t2 - t1;
        timing_acc->dsmem_cycles = t3 - t2;
        timing_acc->sync2_cycles = t4 - t3;
        timing_acc->iterations = 1;
    }
}

// Wrappers for cluster-based combine (select VEC_SIZE based on alignment)

// OPT_MODE: 0 = Original, 1 = Break Dependency Chain, 2 = cp.async

// Wrapper for original implementation (mode 0)
template <int TOP_K, int CLUSTER_SIZE, typename T, bool COLLECT_TIMING>
__device__ void vectorized_combine_cluster_mode0(T* dst_typed_base, int size_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs, uint8_t* smem_partial,
    ClusterTimingAccumulator* timing_acc)
{
    if (size_per_token % 16 == 0)
    {
        vectorized_combine_cluster_impl<16, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else if (size_per_token % 8 == 0)
    {
        vectorized_combine_cluster_impl<8, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else if (size_per_token % 4 == 0)
    {
        vectorized_combine_cluster_impl<4, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else if (size_per_token % 2 == 0)
    {
        vectorized_combine_cluster_impl<2, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else
    {
        vectorized_combine_cluster_impl<1, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
}

// Wrapper for break dependency chain implementation (mode 1)
template <int TOP_K, int CLUSTER_SIZE, typename T, bool COLLECT_TIMING>
__device__ void vectorized_combine_cluster_mode1(T* dst_typed_base, int size_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs, uint8_t* smem_partial,
    ClusterTimingAccumulator* timing_acc)
{
    if (size_per_token % 16 == 0)
    {
        vectorized_combine_cluster_break_dep_impl<16, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else if (size_per_token % 8 == 0)
    {
        vectorized_combine_cluster_break_dep_impl<8, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else if (size_per_token % 4 == 0)
    {
        vectorized_combine_cluster_break_dep_impl<4, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else if (size_per_token % 2 == 0)
    {
        vectorized_combine_cluster_break_dep_impl<2, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
    else
    {
        vectorized_combine_cluster_break_dep_impl<1, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
    }
}

// Wrapper for cp.async implementation (mode 2)
// Note: requires additional shared memory buffer (smem_async_buffers)
template <int TOP_K, int CLUSTER_SIZE, typename T, bool COLLECT_TIMING>
__device__ void vectorized_combine_cluster_mode2(T* dst_typed_base, int size_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs, uint8_t* smem_partial, uint8_t* smem_async_buffers,
    ClusterTimingAccumulator* timing_acc)
{
    if (size_per_token % 16 == 0)
    {
        vectorized_combine_cluster_cpasync_impl<16, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, smem_async_buffers,
            timing_acc);
    }
    else if (size_per_token % 8 == 0)
    {
        vectorized_combine_cluster_cpasync_impl<8, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, smem_async_buffers,
            timing_acc);
    }
    else if (size_per_token % 4 == 0)
    {
        vectorized_combine_cluster_cpasync_impl<4, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, smem_async_buffers,
            timing_acc);
    }
    else if (size_per_token % 2 == 0)
    {
        vectorized_combine_cluster_cpasync_impl<2, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, smem_async_buffers,
            timing_acc);
    }
    else
    {
        vectorized_combine_cluster_cpasync_impl<1, TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, smem_async_buffers,
            timing_acc);
    }
}

// Legacy wrapper for backward compatibility (uses mode 0)
template <int TOP_K, int CLUSTER_SIZE, typename T, bool COLLECT_TIMING>
__device__ void vectorized_combine_cluster(T* dst_typed_base, int size_per_token, int rank_id, int max_tokens_per_rank,
    CombineKernelPointers const& ptrs, uint8_t* smem_partial, ClusterTimingAccumulator* timing_acc)
{
    vectorized_combine_cluster_mode0<TOP_K, CLUSTER_SIZE, T, COLLECT_TIMING>(
        dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, timing_acc);
}

#endif // __CUDA_ARCH__ >= 900

// Cluster-based Combine Kernel
// Uses CLUSTER_SIZE CTAs per token, each CTA fetches TOP_K/CLUSTER_SIZE ranks in parallel
// Requires SM90+ for distributed shared memory support
//
// OPT_MODE: 0 = Original (load + accumulate in same loop)
//           1 = Break dependency chain (load all first, then accumulate)
//           2 = cp.async (hardware async copy to shared memory)
template <typename T, int TOP_K, int CLUSTER_SIZE, int OPT_MODE>
#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
#else
__global__
#endif
    void moeA2ACombineClusterKernel(const CombineKernelPointers ptrs, int max_tokens_per_rank, int elements_per_token,
        int local_num_tokens, int rank_id, int ep_size)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

#if ENABLE_A2A_TIMING_STATS
    uint64_t t_start = 0, t_sync_end = 0, t_data_end = 0;
    bool const is_timing_thread = (cluster.block_rank() == 0 && blockIdx.x == 0 && threadIdx.x == 0);
    if (is_timing_thread)
    {
        t_start = clock64();
    }
#endif

    // Each token uses CLUSTER_SIZE blocks
    int const local_token_idx = blockIdx.x / CLUSTER_SIZE;
    int const size_per_token = elements_per_token * sizeof(T);

    // Early exit for threads beyond token range
    if (local_num_tokens == 0)
    {
        if (local_token_idx > 0)
            return;
    }
    else
    {
        if (local_token_idx >= local_num_tokens)
            return;
    }

#if !DISABLE_SYNC_FOR_PROFILING
    // Synchronization: Only CTA 0 of cluster 0 signals, all CTAs wait
    bool is_first_warp = threadIdx.x / warpSize == 0;
    int const cluster_rank = cluster.block_rank();

    if (is_first_warp)
    {
        int lane_id = threadIdx.x % warpSize;
        uint32_t expected_value = *ptrs.flag_val;

        // Only cluster_rank 0 of block 0 signals
        if (blockIdx.x == 0 && cluster_rank == 0)
        {
#pragma unroll 1
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                uint32_t* flag_addr = &ptrs.completion_flags[peer_rank][rank_id];
                asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));
            }
        }

        // All blocks wait for completion
#pragma unroll 1
        for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
        {
            bool flag_set = false;
            auto s = clock64();
            do
            {
                uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
                uint32_t flag_value;
                asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
                flag_set = flag_value == expected_value;
            } while (!flag_set && !check_timeout(s));

            if (__builtin_expect(!flag_set, 0))
            {
                printf("combine_cluster: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id,
                    peer_rank);
                asm volatile("trap;");
                return;
            }
        }
        asm volatile("fence.acquire.sys;");
    }
    __syncthreads();
#endif // DISABLE_SYNC_FOR_PROFILING

#if ENABLE_A2A_TIMING_STATS
    if (is_timing_thread)
    {
        t_sync_end = clock64();
    }
#endif

    if (local_num_tokens == 0)
        return;

    // Shared memory layout:
    // - Mode 0 & 1: smem_partial[size_per_token] for partial results
    // - Mode 2: smem_partial[size_per_token] + smem_async[RANKS_PER_CTA * size_per_token] for async buffers
    extern __shared__ uint8_t smem_partial[];

    // Get output location for this token
    T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;

    // Timing accumulator for cluster-specific stats
    ClusterTimingAccumulator timing_acc;

    // Perform cluster-based combine based on optimization mode
    if constexpr (OPT_MODE == 0)
    {
        // Mode 0: Original (load + accumulate in same loop)
#if ENABLE_A2A_TIMING_STATS
        vectorized_combine_cluster_mode0<TOP_K, CLUSTER_SIZE, T, true>(
            token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, &timing_acc);
#else
        vectorized_combine_cluster_mode0<TOP_K, CLUSTER_SIZE, T, false>(
            token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, &timing_acc);
#endif
    }
    else if constexpr (OPT_MODE == 1)
    {
        // Mode 1: Break dependency chain (load all first into registers, then accumulate)
#if ENABLE_A2A_TIMING_STATS
        vectorized_combine_cluster_mode1<TOP_K, CLUSTER_SIZE, T, true>(
            token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, &timing_acc);
#else
        vectorized_combine_cluster_mode1<TOP_K, CLUSTER_SIZE, T, false>(
            token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, &timing_acc);
#endif
    }
    else if constexpr (OPT_MODE == 2)
    {
        // Mode 2: cp.async (hardware async copy)
        // Async buffers are placed after smem_partial
        constexpr int RANKS_PER_CTA = (TOP_K + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
        uint8_t* smem_async_buffers = smem_partial + size_per_token;
#if ENABLE_A2A_TIMING_STATS
        vectorized_combine_cluster_mode2<TOP_K, CLUSTER_SIZE, T, true>(
            token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, smem_async_buffers,
            &timing_acc);
#else
        vectorized_combine_cluster_mode2<TOP_K, CLUSTER_SIZE, T, false>(
            token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs, smem_partial, smem_async_buffers,
            &timing_acc);
#endif
    }

#if ENABLE_A2A_TIMING_STATS
    if (is_timing_thread && ptrs.timing_stats != nullptr)
    {
        t_data_end = clock64();
        ptrs.timing_stats[rank_id].combine_sync_cycles = t_sync_end - t_start;
        ptrs.timing_stats[rank_id].combine_data_cycles = t_data_end - t_sync_end;

        // Write cluster-specific breakdown
        ptrs.timing_stats[rank_id].combine_cluster_remote_read_cycles = timing_acc.remote_read_cycles;
        ptrs.timing_stats[rank_id].combine_cluster_sync1_cycles = timing_acc.sync1_cycles;
        ptrs.timing_stats[rank_id].combine_cluster_dsmem_cycles = timing_acc.dsmem_cycles;
        ptrs.timing_stats[rank_id].combine_cluster_sync2_cycles = timing_acc.sync2_cycles;
        ptrs.timing_stats[rank_id].combine_cluster_iterations = timing_acc.iterations;
    }
#endif

#else
    // Fallback for non-SM90 architectures: error
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("ERROR: Cluster-based combine kernel requires SM90+ (Hopper)\n");
    }
#endif // __CUDA_ARCH__ >= 900
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params)
{
    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32; // 8 warps per block

    // Calculate bytes per token based on dtype
    int element_size;
    switch (params.dtype)
    {
    case nvinfer1::DataType::kHALF: element_size = sizeof(half); break;
    case nvinfer1::DataType::kBF16: element_size = sizeof(__nv_bfloat16); break;
    case nvinfer1::DataType::kFLOAT: element_size = sizeof(float); break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported dtype for combine prepare"); return;
    }

    int bytes_per_token = params.elements_per_token * element_size;
    int global_token_num = params.prepare_payload == nullptr ? 1 : params.ep_size * params.max_tokens_per_rank;
    int grid_size_warp = ceilDiv(global_token_num, kWarpsPerBlock);
    int grid_size_block = global_token_num; // one block per token

    if (params.one_block_per_token)
    {
        moeA2APrepareCombineKernel<BlockPolicy><<<grid_size_block, kBlockSize, 0, params.stream>>>(
            static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank])),
            static_cast<uint8_t const*>(params.prepare_payload), bytes_per_token, params.ep_size,
            params.max_tokens_per_rank, params.flag_val, params.recv_counters);
    }
    else
    {
        moeA2APrepareCombineKernel<WarpPolicy><<<grid_size_warp, kBlockSize, 0, params.stream>>>(
            static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank])),
            static_cast<uint8_t const*>(params.prepare_payload), bytes_per_token, params.ep_size,
            params.max_tokens_per_rank, params.flag_val, params.recv_counters);
    }
}

// ============================================================================
// Combine Launch Function
// ============================================================================

// Switch macro for optimization mode
#define SWITCH_OPT_MODE(opt_mode, OPT_MODE, ...)                                                                       \
    switch (opt_mode)                                                                                                  \
    {                                                                                                                  \
    case 0:                                                                                                            \
    {                                                                                                                  \
        constexpr int OPT_MODE = 0;                                                                                    \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 1:                                                                                                            \
    {                                                                                                                  \
        constexpr int OPT_MODE = 1;                                                                                    \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 2:                                                                                                            \
    {                                                                                                                  \
        constexpr int OPT_MODE = 2;                                                                                    \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Unsupported opt_mode (must be 0, 1, or 2)");                                      \
    }                                                                                                                  \
    }

// Helper function to launch cluster-based combine kernel with cudaLaunchKernelEx
template <typename T, int TOP_K, int CLUSTER_SIZE, int OPT_MODE>
void launchClusterCombineKernel(CombineKernelPointers const& kernel_ptrs, MoeA2ACombineParams const& params,
    int grid_size, int block_size, int shared_mem_size)
{
#if __CUDACC_VER_MAJOR__ >= 12
    // Configure launch attributes for cluster
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_size, 1, 1);
    config.blockDim = dim3(block_size, 1, 1);
    config.dynamicSmemBytes = shared_mem_size;
    config.stream = params.stream;

    // Set cluster dimension attribute
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    // Create kernel arguments
    void* kernel_args[] = {const_cast<CombineKernelPointers*>(&kernel_ptrs),
        const_cast<int*>(&params.max_tokens_per_rank), const_cast<int*>(&params.elements_per_token),
        const_cast<int*>(&params.local_num_tokens), const_cast<int*>(&params.ep_rank),
        const_cast<int*>(&params.ep_size)};

    // Launch with cluster configuration
    auto kernel_func = moeA2ACombineClusterKernel<T, TOP_K, CLUSTER_SIZE, OPT_MODE>;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_func, kernel_ptrs, params.max_tokens_per_rank,
        params.elements_per_token, params.local_num_tokens, params.ep_rank, params.ep_size));
#else
    TLLM_CHECK_WITH_INFO(false, "Cluster-based combine kernel requires CUDA 12.0 or higher");
#endif
}

void moe_a2a_combine_launch(MoeA2ACombineParams const& params)
{
    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.local_num_tokens >= 0);
    TLLM_CHECK(params.elements_per_token > 0);

    // Get cluster size from environment variable or params
    // Environment variable takes precedence if set
    int const envClusterSize = tensorrt_llm::common::getEnvMoeA2AClusterSize();
    int const clusterSize = (envClusterSize > 1) ? envClusterSize : params.cluster_size;

    // Validate cluster_size
    if (clusterSize > 1)
    {
        TLLM_CHECK_WITH_INFO(clusterSize == 2 || clusterSize == 4 || clusterSize == 8,
            "cluster_size must be 1 (disabled), 2, 4, or 8");
        TLLM_CHECK_WITH_INFO(
            params.one_block_per_token, "Cluster-based combine requires one_block_per_token=true (BlockPolicy)");
    }

    // Check if timing stats is enabled via environment variable
    bool const enableTimingStats = tensorrt_llm::common::getEnvMoeA2ATimingStats();
    MoeA2ATimingStats* deviceTimingStats = nullptr;
    if (enableTimingStats)
    {
        // Allocate device buffer for timing stats (one per rank)
        TLLM_CUDA_CHECK(cudaMallocAsync(&deviceTimingStats, sizeof(MoeA2ATimingStats) * params.ep_size, params.stream));
        TLLM_CUDA_CHECK(
            cudaMemsetAsync(deviceTimingStats, 0, sizeof(MoeA2ATimingStats) * params.ep_size, params.stream));
    }

    // Configure kernel launch
    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ACombineBlockSize();
    int const kWarpsPerBlock = kBlockSize / 32; // warpSize
    int grid_size_warp = ceilDiv(params.local_num_tokens, kWarpsPerBlock);
    int grid_size_block = params.local_num_tokens;
    // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
    if (grid_size_warp == 0)
    {
        grid_size_warp = 1;
    }
    if (grid_size_block == 0)
    {
        grid_size_block = 1;
    }

    // Prepare kernel pointers struct for combine
    CombineKernelPointers kernel_ptrs = {}; // Zero-initialize

    // Set output data pointer in src_data_ptrs[0]
    kernel_ptrs.src_data_ptrs[0] = params.output_data;

    // Fill recv buffer pointers
    for (int rank = 0; rank < params.ep_size; rank++)
    {
        kernel_ptrs.recv_buffers[rank][0] = params.recv_buffers[rank];
    }

    // Copy completion flag pointers
    for (int i = 0; i < params.ep_size; i++)
    {
        kernel_ptrs.completion_flags[i] = params.completion_flags[i];
    }
    kernel_ptrs.flag_val = params.flag_val;

    // Copy communication tracking pointers
    kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
    kernel_ptrs.topk_send_indices = params.topk_send_indices;

    // Copy timing stats pointer (use device buffer if env var enabled, otherwise use params)
    kernel_ptrs.timing_stats = enableTimingStats ? deviceTimingStats : params.timing_stats;

    // Choose between cluster-based and regular kernel
    if (clusterSize > 1)
    {
        // Cluster-based combine kernel (SM90+ with DSMEM)
        // Grid size: local_num_tokens * cluster_size (each token uses cluster_size blocks)
        int grid_size_cluster = params.local_num_tokens * clusterSize;
        if (grid_size_cluster == 0)
        {
            grid_size_cluster = clusterSize; // Minimal launch for sync
        }

        // Get optimization mode from environment variable
        // 0 = Original, 1 = Break dependency chain, 2 = cp.async
        int const optMode = tensorrt_llm::common::getEnvMoeA2AClusterOptMode();

        // Calculate shared memory size based on optimization mode
        // For BF16 with hidden_size=7168: 7168 * 2 = 14336 bytes = 14KB
        // H100/H200: 228KB smem available, B200/B300: ~256KB - plenty of room
        int element_size;
        switch (params.dtype)
        {
        case nvinfer1::DataType::kHALF: element_size = sizeof(half); break;
        case nvinfer1::DataType::kBF16: element_size = sizeof(__nv_bfloat16); break;
        case nvinfer1::DataType::kFLOAT: element_size = sizeof(float); break;
        default: element_size = 2; break;
        }
        int const size_per_token = params.elements_per_token * element_size;

        // Shared memory layout:
        // - Mode 0 & 1: smem_partial[size_per_token]
        // - Mode 2: smem_partial[size_per_token] + smem_async[RANKS_PER_CTA * size_per_token]
        //   where RANKS_PER_CTA = ceil(TOP_K / CLUSTER_SIZE)
        int shared_mem_size = size_per_token;
        if (optMode == 2)
        {
            // For cp.async mode, need additional buffer for async copies
            int const ranks_per_cta = (params.top_k + clusterSize - 1) / clusterSize;
            shared_mem_size += ranks_per_cta * size_per_token;
        }

        SWITCH_DTYPE(params.dtype, TKernelType, {
            SWITCH_TOP_K(params.top_k, TOP_K, {
                SWITCH_CLUSTER_SIZE(clusterSize, CLUSTER_SIZE, {
                    SWITCH_OPT_MODE(optMode, OPT_MODE, {
                        launchClusterCombineKernel<TKernelType, TOP_K, CLUSTER_SIZE, OPT_MODE>(
                            kernel_ptrs, params, grid_size_cluster, kBlockSize, shared_mem_size);
                    });
                });
            });
        });
    }
    else
    {
        // Regular combine kernel (no cluster)
        SWITCH_DTYPE(params.dtype, TKernelType, {
            SWITCH_POLICY(params.one_block_per_token, Policy, {
                SWITCH_TOP_K(params.top_k, TOP_K, {
                    auto launch = [&](int grid_blocks, int block_threads)
                    {
                        moeA2ACombineKernel<TKernelType, Policy, TOP_K>
                            <<<grid_blocks, block_threads, 0, params.stream>>>(kernel_ptrs, params.max_tokens_per_rank,
                                params.elements_per_token, params.local_num_tokens, params.ep_rank, params.ep_size);
                    };
                    int grid = params.one_block_per_token ? grid_size_block : grid_size_warp;
                    int cta = kBlockSize;
                    launch(grid, cta);
                });
            });
        });
    }

    // If timing stats enabled via env var, sync, copy back to host, and print
    if (enableTimingStats && deviceTimingStats != nullptr)
    {
        TLLM_CUDA_CHECK(cudaStreamSynchronize(params.stream));
        std::vector<MoeA2ATimingStats> hostTimingStats(params.ep_size);
        TLLM_CUDA_CHECK(cudaMemcpy(hostTimingStats.data(), deviceTimingStats,
            sizeof(MoeA2ATimingStats) * params.ep_size, cudaMemcpyDeviceToHost));

        // Print combine timing for this rank only
        float const gpuFreqGhz = tensorrt_llm::common::getEnvMoeA2AGpuFreqGhz();
        float const cyclesToUs = 1.0f / (gpuFreqGhz * 1000.0f);
        MoeA2ATimingStats const& s = hostTimingStats[params.ep_rank];
        float combineSync = s.combine_sync_cycles * cyclesToUs;
        float combineData = s.combine_data_cycles * cyclesToUs;
        float combineTotal = combineSync + combineData;

        // Get optimization mode for printing
        int const printOptMode = (clusterSize > 1) ? tensorrt_llm::common::getEnvMoeA2AClusterOptMode() : -1;
        char const* optModeNames[] = {"original", "break-dep", "cp.async"};
        char const* optModeName = (printOptMode >= 0 && printOptMode <= 2) ? optModeNames[printOptMode] : "N/A";

        printf("[MoE A2A Combine Timing] Rank %d (tokens=%d, cluster=%d, opt=%s): Sync=%.2f us (%.1f%%), Data=%.2f us "
               "(%.1f%%), "
               "Total=%.2f us\n",
            params.ep_rank, params.local_num_tokens, clusterSize, optModeName, combineSync,
            combineSync / combineTotal * 100, combineData, combineData / combineTotal * 100, combineTotal);

        // Print cluster breakdown if available
        if (clusterSize > 1 && s.combine_cluster_iterations > 0)
        {
            float clusterRemote = s.combine_cluster_remote_read_cycles * cyclesToUs;
            float clusterSync1 = s.combine_cluster_sync1_cycles * cyclesToUs;
            float clusterDsmem = s.combine_cluster_dsmem_cycles * cyclesToUs;
            float clusterSync2 = s.combine_cluster_sync2_cycles * cyclesToUs;
            float clusterTotal = clusterRemote + clusterSync1 + clusterDsmem + clusterSync2;
            int iters = s.combine_cluster_iterations;

            printf("[MoE A2A Cluster Breakdown] iters=%d: RemoteRead=%.2f us (%.2f/iter), "
                   "Sync1=%.2f us (%.2f/iter), DSMEM=%.2f us (%.2f/iter), Sync2=%.2f us (%.2f/iter), "
                   "TotalSync=%.2f us (%.1f%%)\n",
                iters, clusterRemote, clusterRemote / iters, clusterSync1, clusterSync1 / iters, clusterDsmem,
                clusterDsmem / iters, clusterSync2, clusterSync2 / iters, clusterSync1 + clusterSync2,
                (clusterSync1 + clusterSync2) / clusterTotal * 100);
        }

        TLLM_CUDA_CHECK(cudaFreeAsync(deviceTimingStats, params.stream));
    }
}

// Kernel to sanitize expert ids for invalid tokens
__global__ void moeA2ASanitizeExpertIdsKernel(int32_t* expert_ids_ptr, int32_t const* recv_counters_ptr, int ep_size,
    int max_tokens_per_rank, int top_k, int32_t invalid_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = ep_size * max_tokens_per_rank;
    if (tid >= total_tokens)
        return;

    int source_rank = tid / max_tokens_per_rank;
    int token_idx = tid % max_tokens_per_rank;

    if (token_idx >= recv_counters_ptr[source_rank])
    {
        int32_t* token_expert_ids = expert_ids_ptr + tid * top_k;
        for (int k = 0; k < top_k; ++k)
        {
            token_expert_ids[k] = invalid_id;
        }
    }
}

void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, int32_t invalid_id,
    int ep_size, int max_tokens_per_rank, int top_k, cudaStream_t stream)
{
    constexpr int kBlockSize = 256;
    int total_tokens = ep_size * max_tokens_per_rank;
    int grid = ceilDiv(total_tokens, kBlockSize);
    moeA2ASanitizeExpertIdsKernel<<<grid, kBlockSize, 0, stream>>>(
        expert_ids, recv_counters, ep_size, max_tokens_per_rank, top_k, invalid_id);
}

void moe_a2a_print_timing_stats(MoeA2ATimingStats const* host_stats, int ep_size, float gpu_freq_ghz)
{
    if (host_stats == nullptr)
    {
        printf("[MoE A2A Timing] No timing stats available (host_stats is null)\n");
        return;
    }

    // Accumulate totals across all ranks
    uint64_t total_dispatch_routing = 0;
    uint64_t total_dispatch_data = 0;
    uint64_t total_dispatch_sync = 0;
    uint64_t total_combine_sync = 0;
    uint64_t total_combine_data = 0;

    // Cluster-specific totals
    uint64_t total_cluster_remote_read = 0;
    uint64_t total_cluster_sync1 = 0;
    uint64_t total_cluster_dsmem = 0;
    uint64_t total_cluster_sync2 = 0;
    int total_cluster_iterations = 0;
    bool has_cluster_stats = false;

    printf("\n");
    printf("============================================================\n");
    printf(" MoE All-to-All Timing Statistics (GPU cycles → microseconds)\n");
    printf(" GPU Frequency: %.2f GHz\n", gpu_freq_ghz);
    printf("============================================================\n");

    // Print per-rank stats
    for (int r = 0; r < ep_size; r++)
    {
        MoeA2ATimingStats const& s = host_stats[r];
        float cycles_to_us = 1.0f / (gpu_freq_ghz * 1000.0f); // cycles to microseconds

        float dispatch_routing_us = s.dispatch_routing_cycles * cycles_to_us;
        float dispatch_data_us = s.dispatch_data_cycles * cycles_to_us;
        float dispatch_sync_us = s.dispatch_sync_cycles * cycles_to_us;
        float combine_sync_us = s.combine_sync_cycles * cycles_to_us;
        float combine_data_us = s.combine_data_cycles * cycles_to_us;

        float dispatch_total_us = dispatch_routing_us + dispatch_data_us + dispatch_sync_us;
        float combine_total_us = combine_sync_us + combine_data_us;
        float total_us = dispatch_total_us + combine_total_us;

        printf("\n--- Rank %d (tokens: %d) ---\n", s.rank_id, s.local_num_tokens);
        printf("  Dispatch:\n");
        printf("    Routing:     %10.2f us (%5.1f%%)\n", dispatch_routing_us, dispatch_routing_us / total_us * 100);
        printf("    Data:        %10.2f us (%5.1f%%)\n", dispatch_data_us, dispatch_data_us / total_us * 100);
        printf("    Sync:        %10.2f us (%5.1f%%)\n", dispatch_sync_us, dispatch_sync_us / total_us * 100);
        printf("    Subtotal:    %10.2f us\n", dispatch_total_us);
        printf("  Combine:\n");
        printf("    Sync:        %10.2f us (%5.1f%%)\n", combine_sync_us, combine_sync_us / total_us * 100);
        printf("    Data:        %10.2f us (%5.1f%%)\n", combine_data_us, combine_data_us / total_us * 100);
        printf("    Subtotal:    %10.2f us\n", combine_total_us);
        printf("  Total:         %10.2f us\n", total_us);

        // Print cluster-specific breakdown if available
        if (s.combine_cluster_iterations > 0)
        {
            has_cluster_stats = true;
            float cluster_remote_us = s.combine_cluster_remote_read_cycles * cycles_to_us;
            float cluster_sync1_us = s.combine_cluster_sync1_cycles * cycles_to_us;
            float cluster_dsmem_us = s.combine_cluster_dsmem_cycles * cycles_to_us;
            float cluster_sync2_us = s.combine_cluster_sync2_cycles * cycles_to_us;
            float cluster_total_us = cluster_remote_us + cluster_sync1_us + cluster_dsmem_us + cluster_sync2_us;

            printf("  Cluster Breakdown (iterations=%d):\n", s.combine_cluster_iterations);
            printf("    Remote Read: %10.2f us (%5.1f%%) [%.2f us/iter]\n", cluster_remote_us,
                cluster_remote_us / cluster_total_us * 100, cluster_remote_us / s.combine_cluster_iterations);
            printf("    Sync1:       %10.2f us (%5.1f%%) [%.2f us/iter]\n", cluster_sync1_us,
                cluster_sync1_us / cluster_total_us * 100, cluster_sync1_us / s.combine_cluster_iterations);
            printf("    DSMEM:       %10.2f us (%5.1f%%) [%.2f us/iter]\n", cluster_dsmem_us,
                cluster_dsmem_us / cluster_total_us * 100, cluster_dsmem_us / s.combine_cluster_iterations);
            printf("    Sync2:       %10.2f us (%5.1f%%) [%.2f us/iter]\n", cluster_sync2_us,
                cluster_sync2_us / cluster_total_us * 100, cluster_sync2_us / s.combine_cluster_iterations);
            printf("    Cluster Tot: %10.2f us\n", cluster_total_us);

            total_cluster_remote_read += s.combine_cluster_remote_read_cycles;
            total_cluster_sync1 += s.combine_cluster_sync1_cycles;
            total_cluster_dsmem += s.combine_cluster_dsmem_cycles;
            total_cluster_sync2 += s.combine_cluster_sync2_cycles;
            total_cluster_iterations += s.combine_cluster_iterations;
        }

        total_dispatch_routing += s.dispatch_routing_cycles;
        total_dispatch_data += s.dispatch_data_cycles;
        total_dispatch_sync += s.dispatch_sync_cycles;
        total_combine_sync += s.combine_sync_cycles;
        total_combine_data += s.combine_data_cycles;
    }

    // Print average across ranks
    float cycles_to_us = 1.0f / (gpu_freq_ghz * 1000.0f);
    float avg_dispatch_routing_us = (total_dispatch_routing / ep_size) * cycles_to_us;
    float avg_dispatch_data_us = (total_dispatch_data / ep_size) * cycles_to_us;
    float avg_dispatch_sync_us = (total_dispatch_sync / ep_size) * cycles_to_us;
    float avg_combine_sync_us = (total_combine_sync / ep_size) * cycles_to_us;
    float avg_combine_data_us = (total_combine_data / ep_size) * cycles_to_us;

    float avg_dispatch_total_us = avg_dispatch_routing_us + avg_dispatch_data_us + avg_dispatch_sync_us;
    float avg_combine_total_us = avg_combine_sync_us + avg_combine_data_us;
    float avg_total_us = avg_dispatch_total_us + avg_combine_total_us;

    printf("\n============================================================\n");
    printf(" AVERAGE ACROSS %d RANKS\n", ep_size);
    printf("============================================================\n");
    printf("  Dispatch Routing:  %10.2f us (%5.1f%%)\n", avg_dispatch_routing_us,
        avg_dispatch_routing_us / avg_total_us * 100);
    printf("  Dispatch Data:     %10.2f us (%5.1f%%)\n", avg_dispatch_data_us,
        avg_dispatch_data_us / avg_total_us * 100);
    printf("  Dispatch Sync:     %10.2f us (%5.1f%%)\n", avg_dispatch_sync_us,
        avg_dispatch_sync_us / avg_total_us * 100);
    printf("  Combine Sync:      %10.2f us (%5.1f%%)\n", avg_combine_sync_us,
        avg_combine_sync_us / avg_total_us * 100);
    printf("  Combine Data:      %10.2f us (%5.1f%%)\n", avg_combine_data_us,
        avg_combine_data_us / avg_total_us * 100);
    printf("------------------------------------------------------------\n");
    printf("  Dispatch Total:    %10.2f us\n", avg_dispatch_total_us);
    printf("  Combine Total:     %10.2f us\n", avg_combine_total_us);
    printf("  TOTAL:             %10.2f us\n", avg_total_us);

    // Print cluster average if available
    if (has_cluster_stats && total_cluster_iterations > 0)
    {
        int avg_iterations = total_cluster_iterations / ep_size;
        float avg_cluster_remote_us = (total_cluster_remote_read / ep_size) * cycles_to_us;
        float avg_cluster_sync1_us = (total_cluster_sync1 / ep_size) * cycles_to_us;
        float avg_cluster_dsmem_us = (total_cluster_dsmem / ep_size) * cycles_to_us;
        float avg_cluster_sync2_us = (total_cluster_sync2 / ep_size) * cycles_to_us;
        float avg_cluster_total_us = avg_cluster_remote_us + avg_cluster_sync1_us + avg_cluster_dsmem_us + avg_cluster_sync2_us;

        printf("\n============================================================\n");
        printf(" CLUSTER COMBINE BREAKDOWN (avg iterations=%d)\n", avg_iterations);
        printf("============================================================\n");
        printf("  Remote Read:   %10.2f us (%5.1f%%) [%.2f us/iter]\n", avg_cluster_remote_us,
            avg_cluster_remote_us / avg_cluster_total_us * 100, avg_cluster_remote_us / avg_iterations);
        printf("  Sync1:         %10.2f us (%5.1f%%) [%.2f us/iter]\n", avg_cluster_sync1_us,
            avg_cluster_sync1_us / avg_cluster_total_us * 100, avg_cluster_sync1_us / avg_iterations);
        printf("  DSMEM Gather:  %10.2f us (%5.1f%%) [%.2f us/iter]\n", avg_cluster_dsmem_us,
            avg_cluster_dsmem_us / avg_cluster_total_us * 100, avg_cluster_dsmem_us / avg_iterations);
        printf("  Sync2:         %10.2f us (%5.1f%%) [%.2f us/iter]\n", avg_cluster_sync2_us,
            avg_cluster_sync2_us / avg_cluster_total_us * 100, avg_cluster_sync2_us / avg_iterations);
        printf("------------------------------------------------------------\n");
        printf("  Cluster Total: %10.2f us\n", avg_cluster_total_us);
        printf("  Total Sync:    %10.2f us (%5.1f%%)\n", avg_cluster_sync1_us + avg_cluster_sync2_us,
            (avg_cluster_sync1_us + avg_cluster_sync2_us) / avg_cluster_total_us * 100);
    }

    printf("============================================================\n\n");
}

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
