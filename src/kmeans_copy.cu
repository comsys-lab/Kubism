#include <cassert>
#include <cstdio>
#include <cfloat>
#include <cinttypes>
#include <algorithm>
#include <memory>
#include <time.h>
#include <curand_kernel.h>
#include <iostream>
#include <atomic>

#include <thread>
#include <chrono>
#include <iostream>

#include "private.h"
#include "metric_abstraction.h"
#include "tricks.cuh"

#include "kmeans_cpu.h"





//#define SAMPLES_SIZE 1000000


#define BS_KMPP 1024
#define BS_AFKMC2_Q 512
#define BS_AFKMC2_R 512
#define BS_AFKMC2_MDT 512
#define BS_LL_ASS 128
#define BS_LL_CNT 256
#define BS_YY_INI 128
#define BS_YY_GFL 512
#define BS_YY_LFL 512
#define BLOCK_SIZE 1024  // for all the rest of the kernels
#define SHMEM_AFKMC2_RC 8191  // in float-s, the actual value is +1
#define SHMEM_AFKMC2_MT 8192

#define YINYANG_GROUP_TOLERANCE 0.02
#define YINYANG_DRAFT_REASSIGNMENTS 0.11
#define YINYANG_REFRESH_EPSILON 1e-4

__device__ uint32_t d_changed_number;
__device__ uint32_t d_passed_number;
__device__ uint32_t d_distance_number;
__device__ uint32_t d_mark_gpu_number;
__device__ uint32_t d_mark_cpu_number;


__constant__ uint32_t d_samples_size;
__constant__ uint32_t d_clusters_size;
__constant__ uint32_t d_yy_groups_size;
__constant__ int d_shmem_size;



// __constant__ uint32_t threshold;
// __device__ uint32_t d_cpu_changed_number;

//__device__ uint32_t d_cnt[1025];

//__device__ uint32_t d_cal[1024*1000000];

//__device__ int d_cnt[1000000] = {0};
//__device__ int d_col[1000000] = {0};


//////////////////////----------------------------------------------------------
// Device functions //----------------------------------------------------------
//////////////////////----------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_plus_plus(
    const uint32_t offset, const uint32_t length, const uint32_t cc,
    const F *__restrict__ samples, const F *__restrict__ centroids,
    float *__restrict__ dists, atomic_float *__restrict__ dists_sum) {
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    centroids += (cc - 1) * d_features_size;
    const uint32_t local_sample = sample + offset;
    if (_eq(samples[local_sample], samples[local_sample])) {
      dist = METRIC<M, F>::distance_t(
          samples, centroids, d_samples_size, local_sample);
    }
    float prev_dist;
    if (cc == 1 || dist < (prev_dist = dists[sample])) {
      dists[sample] = dist;
    } else {
      dist = prev_dist;
    }
  }
  dist = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(dists_sum, dist);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_calc_q_dists(
    const uint32_t offset, const uint32_t length, uint32_t c1_index,
    const F *__restrict__ samples, float *__restrict__ dists,
    atomic_float *__restrict__ dsum) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    sample += offset;
    extern __shared__ float shmem_afkmc2[];
    auto c1 = reinterpret_cast<F*>(shmem_afkmc2);
    uint16_t size_each = dupper(d_features_size, static_cast<uint16_t>(blockDim.x));
    for (uint16_t i = size_each * threadIdx.x;
         i < min(size_each * (threadIdx.x + 1), d_features_size); i++) {
      c1[i] = samples[static_cast<uint64_t>(c1_index) * d_features_size + i];
    }
    __syncthreads();
    dist = METRIC<M, F>::distance_t(samples, c1, d_samples_size, sample);
    dist *= dist;
    dists[sample] = dist;
  }
  float sum = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(dsum, sum);
  }
}

__global__ void kmeans_afkmc2_calc_q(
    const uint32_t offset, const uint32_t length,
    float dsum, float *__restrict__ q) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  sample += offset;
  q[sample] = 1 / (2.f * d_samples_size) + q[sample] / (2 * dsum);
}

__global__ void kmeans_afkmc2_random_step(
    const uint32_t m, const uint64_t seed, const uint64_t seq,
    const float *__restrict__ q, uint32_t *__restrict__ choices,
    float *__restrict__ samples) {
  volatile uint32_t ti = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t state;
  curand_init(seed, ti, seq, &state);
  float part = curand_uniform(&state);
  if (ti < m) {
    samples[ti] = curand_uniform(&state);
  }
  float accum = 0, corr = 0;
  bool found = false;
  __shared__ float shared_q[SHMEM_AFKMC2_RC + 1];
  int32_t *all_found = reinterpret_cast<int32_t*>(shared_q + SHMEM_AFKMC2_RC);
  *all_found = blockDim.x;
  const uint32_t size_each = dupper(
      static_cast<unsigned>(SHMEM_AFKMC2_RC), blockDim.x);
  for (uint32_t sample = 0; sample < d_samples_size; sample += SHMEM_AFKMC2_RC) {
    __syncthreads();
    if (*all_found == 0) {
      return;
    }
    for (uint32_t i = 0, si = threadIdx.x * size_each;
         i < size_each && (si = threadIdx.x * size_each + i) < SHMEM_AFKMC2_RC
         && (sample + si) < d_samples_size;
         i++) {
      shared_q[si] = q[sample + si];
    }
    __syncthreads();
    if (!found) {
      int i = 0;
      #pragma unroll 4
      for (; i < SHMEM_AFKMC2_RC && accum < part && sample + i < d_samples_size;
           i++) {
        // Kahan summation with inverted c
        float y = _add(corr, shared_q[i]);
        float t = accum + y;
        corr = y - (t - accum);
        accum = t;
      }
      if (accum >= part) {
        if (ti < m) {
          choices[ti] = sample + i - 1;
        }
        found = true;
        atomicSub(all_found, 1);
      }
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_min_dist(
    const uint32_t m, const uint32_t k, const F *__restrict__ samples,
    const uint32_t *__restrict__ choices, const F *__restrict__ centroids,
    float *__restrict__ min_dists) {
  uint32_t chi = blockIdx.x * blockDim.x + threadIdx.x;
  if (chi >= m) {
    return;
  }
  float min_dist = FLT_MAX;
  for (uint32_t c = 0; c < k; c++) {
    float dist = METRIC<M, F>::distance_t(
        samples, centroids + c * d_features_size, d_samples_size, choices[chi]);
    if (dist < min_dist) {
      min_dist = dist;
    }
  }
  min_dists[chi] = min_dist * min_dist;
}

// min_dists must be set to FLT_MAX or +inf or NAN!
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_afkmc2_min_dist_transposed(
    const uint32_t m, const uint32_t k, const F *__restrict__ samples,
    const uint32_t *__restrict__ choices, const F *__restrict__ centroids,
    float *__restrict__ min_dists) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float shared_min_dists[];
  uint32_t size_each = dupper(m, blockDim.x);
  for (uint32_t i = size_each * threadIdx.x;
       i < min(size_each * (threadIdx.x + 1), m);
       i++) {
    shared_min_dists[i] = FLT_MAX;
  }
  __syncthreads();
  for (uint32_t chi = 0; chi < m; chi++) {
    float dist = FLT_MAX;
    if (c < k) {
      dist = METRIC<M, F>::distance_t(
          samples, centroids + c * d_features_size, d_samples_size, choices[chi]);
    }
    float warp_min = warpReduceMin(dist);
    warp_min *= warp_min;
    if (threadIdx.x % 32 == 0 && c < k) {
      atomicMin(shared_min_dists + chi, warp_min);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t chi = 0; chi < m; chi++) {
      atomicMin(min_dists + chi, shared_min_dists[chi]);
    }
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_assign_lloyd_smallc(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  using HF = typename HALF<F>::type;
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  HF min_dist = _fmax<HF>();
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float _shared_samples[];
  F *shared_samples = reinterpret_cast<F *>(_shared_samples);
  F *shared_centroids = shared_samples + blockDim.x * d_features_size;
  const uint32_t cstep = (d_shmem_size - blockDim.x * d_features_size) /
      (d_features_size + 1);
  F *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;
  const uint32_t local_sample = sample + offset;
  bool insane = _neq(samples[local_sample], samples[local_sample]);
  const uint32_t soffset = threadIdx.x * d_features_size;
  if (!insane) {
    for (uint64_t f = 0; f < d_features_size; f++) {
      shared_samples[soffset + f] = samples[f * d_samples_size + local_sample];
    }
  }

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        csqrs[ci] = METRIC<M, F>::sum_squares(
            centroids + global_offset, shared_centroids + local_offset);
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      F product = _const<F>(0), corr = _const<F>(0);
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (int f = 0; f < d_features_size; f++) {
        F y = _fma(corr, shared_samples[soffset + f], shared_centroids[coffset + f]);
        F t = _add(product, y);
        corr = _sub(y, _sub(t, product));
        product = t;
      }
      HF dist = METRIC<M, F>::distance(_const<F>(0), csqrs[c - gc], product);
      if (_lt(dist, min_dist)) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  if (nearest == UINT32_MAX) {
    if (!insane) {
      printf("CUDA kernel kmeans_assign: nearest neighbor search failed for " "sample %" PRIu32 "\n", sample);
      return;
    } else {
      nearest = d_clusters_size;
    }
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_assign_lloyd(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, uint32_t *__restrict__ assignments_prev,
    uint32_t * __restrict__ assignments) {
  using HF = typename HALF<F>::type;
  uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  HF min_dist = _fmax<HF>();
  uint32_t nearest = UINT32_MAX;
  extern __shared__ float _shared_centroids[];
  F *shared_centroids = reinterpret_cast<F *>(_shared_centroids);
  const uint32_t cstep = d_shmem_size / (d_features_size + 1);
  F *csqrs = shared_centroids + cstep * d_features_size;
  const uint32_t size_each = cstep /
      min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;
  const uint32_t local_sample = sample + offset;
  bool insane = _neq(samples[local_sample], samples[local_sample]);

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        csqrs[ci] = METRIC<M, F>::sum_squares(
            centroids + global_offset, shared_centroids + local_offset);
      }
    }
    __syncthreads();
    if (insane) {
      continue;
    }
    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      F product = _const<F>(0), corr = _const<F>(0);
      coffset = (c - gc) * d_features_size;
      #pragma unroll 4
      for (uint64_t f = 0; f < d_features_size; f++) {
        F y = _fma(corr,
                   samples[static_cast<uint64_t>(d_samples_size) * f + local_sample],
                   shared_centroids[coffset + f]);
        F t = _add(product, y);
        corr = _sub(y, _sub(t, product));
        product = t;
      }
      HF dist = METRIC<M, F>::distance(_const<F>(0), csqrs[c - gc], product);
      if (_lt(dist, min_dist)) {
        min_dist = dist;
        nearest = c;
      }
    }
  }
  if (nearest == UINT32_MAX) {
    if (!insane) {
      printf("CUDA kernel kmeans_assign: nearest neighbor search failed for " "sample %" PRIu32 "\n", sample);
      return;
    } else {
      nearest = d_clusters_size;
    }
  }
  uint32_t ass = assignments[sample];
  assignments_prev[sample] = ass;
  if (ass != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_adjust(
    const uint32_t coffset, const uint32_t length,
    const F *__restrict__ samples,
    const uint32_t *__restrict__ assignments_prev,
    const uint32_t *__restrict__ assignments,
    F *__restrict__ centroids, uint32_t *__restrict__ ccounts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= length) {
    return;
  }
  c += coffset;
  uint32_t my_count = ccounts[c];
  {
    F fmy_count = _const<F>(my_count);
    centroids += c * d_features_size;
    for (int f = 0; f < d_features_size; f++) {
      centroids[f] = _mul(centroids[f], fmy_count);
    }
  }
  extern __shared__ uint32_t ass[];
  int step = d_shmem_size / 2;
  F corr = _const<F>(0);
  for (uint32_t sbase = 0; sbase < d_samples_size; sbase += step) {
    __syncthreads();
    if (threadIdx.x == 0) {
      int pos = sbase;
      for (int i = 0; i < step && sbase + i < d_samples_size; i++) {
        ass[2 * i] = assignments[pos + i];
        ass[2 * i + 1] = assignments_prev[pos + i];
      }
    }
    __syncthreads();
    for (int i = 0; i < step && sbase + i < d_samples_size; i++) {
      uint32_t this_ass = ass[2 * i];
      uint32_t  prev_ass = ass[2 * i + 1];
      int sign = 0;
      if (prev_ass == c && this_ass != c) {
        sign = -1;
        my_count--;
      } else if (prev_ass != c && this_ass == c) {
        sign = 1;
        my_count++;
      }
      if (sign != 0) {
        F fsign = _const<F>(sign);
        #pragma unroll 4
        for (uint64_t f = 0; f < d_features_size; f++) {
          F centroid = centroids[f];
          F y = _fma(corr,
                     samples[static_cast<uint64_t>(d_samples_size) * f + sbase + i],
                     fsign);
          F t = _add(centroid, y);
          corr = _sub(y, _sub(t, centroid));
          centroids[f] = t;
        }
      }
    }
  }
  // my_count can be 0 => we get NaN with L2 and never use this cluster again
  // this is a feature, not a bug
  METRIC<M, F>::normalize(my_count, centroids);
  ccounts[c] = my_count;
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_init(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    const uint32_t *__restrict__ groups, float *__restrict__ volatile bounds) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  //uint32_t cnt = 0;
  for (uint32_t i = 0; i < d_yy_groups_size + 1; i++) {
    bounds[static_cast<uint64_t>(length) * i + sample] = FLT_MAX;
    //cnt++;
  }

  uint32_t nearest = assignments[sample];
  extern __shared__ float shared_memory[];
  F *volatile shared_centroids = reinterpret_cast<F*>(shared_memory);
  const uint32_t cstep = d_shmem_size / d_features_size;  //  96
  const uint32_t size_each = cstep / min(blockDim.x, length - blockIdx.x * blockDim.x) + 1;


  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          shared_centroids[local_offset + f] = centroids[global_offset + f];
        }
      }
    }
    __syncthreads();

    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      float dist = METRIC<M, F>::distance_t(
          samples, shared_centroids + (c - gc) * d_features_size,
          d_samples_size, sample + offset);
      //cnt++;
      if (c != nearest) {
        uint64_t gindex = static_cast<uint64_t>(length) * (1 + group) + sample;
        if (dist < bounds[gindex]) {
          bounds[gindex] = dist;
          //cnt++;
        }
      } else {
        bounds[sample] = dist;
        //cnt++;
      }
    }
  }
  //printf("cnt = %u\n", cnt);
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_calc_drifts(
    const uint32_t offset, const uint32_t length,
    const F *__restrict__ centroids, F *__restrict__ drifts) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= length) {
    return;
  }
  c += offset;
  uint32_t coffset = c * d_features_size;
  (reinterpret_cast<float *>(drifts))[d_clusters_size * d_features_size + c] =
      METRIC<M, F>::distance(centroids + coffset, drifts + coffset);
}

__global__ void kmeans_yy_find_group_max_drifts(
    const uint32_t offset, const uint32_t length,
    const uint32_t *__restrict__ groups, float *__restrict__ drifts) {
  uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= length) {
    return;
  }
  group += offset;
  const uint32_t doffset = d_clusters_size * d_features_size;
  const uint32_t step = d_shmem_size / 2;
  const uint32_t size_each = d_shmem_size /
      (2 * min(blockDim.x, length - blockIdx.x * blockDim.x));
  extern __shared__ uint32_t shmem[];
  float *cd = (float *)shmem;
  uint32_t *cg = shmem + step;
  float my_max = -FLT_MAX;
  for (uint32_t offset = 0; offset < d_clusters_size; offset += step) {
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t local_offset = threadIdx.x * size_each + i;
      uint32_t global_offset = offset + local_offset;
      if (global_offset < d_clusters_size && local_offset < step) {
        cd[local_offset] = drifts[doffset + global_offset];
        cg[local_offset] = groups[global_offset];
      }
    }
    __syncthreads();
    for (uint32_t i = 0; i < step; i++) {
      if (cg[i] == group) {
        float d = cd[i];
        if (my_max < d) {
          my_max = d;
        }
      }
    }
  }
  drifts[group] = my_max;
}

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_global_filter(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ groups,
    const float *__restrict__ drifts, const uint32_t *__restrict__ assignments,
    uint32_t *__restrict__ assignments_prev, float *__restrict__ bounds,
    uint32_t *__restrict__ passed) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= length) {
    return;
  }
  uint32_t cluster = assignments[sample];
  assignments_prev[sample] = cluster;

  float upper_bound = bounds[sample];
  uint32_t doffset = d_clusters_size * d_features_size;
  float cluster_drift = drifts[doffset + cluster];

  upper_bound += cluster_drift;
  float min_lower_bound = FLT_MAX;

  /* Find Global lower bound */
  for (uint32_t g = 0; g < d_yy_groups_size; g++) {
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + g) + sample;
    float lower_bound = bounds[gindex] - drifts[g];
    bounds[gindex] = lower_bound;
    if (lower_bound < min_lower_bound) {
      min_lower_bound = lower_bound;
    }
  }

  // Group filter #1 // 
  if (min_lower_bound >= upper_bound) {
    bounds[sample] = upper_bound;

    return;
  }

  /* Upper bound adjust */
  upper_bound = 0;
  upper_bound = METRIC<M, F>::distance_t( samples, centroids + cluster * d_features_size, d_samples_size, sample + offset);
  bounds[sample] = upper_bound;
  
  // Group filter #2 // 
  if (min_lower_bound >= upper_bound) {
   return;  
  }

  // d'oh!
  passed[atomicAggInc(&d_passed_number)] = sample;

  
}



//------------------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter1(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const uint32_t *__restrict__ passed, const F *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads, uint32_t *__restrict__ calculate_data_point, uint32_t *__restrict__ calculate_centroid){
  
  volatile uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample_idx >= d_passed_number){ 
    return;
  }
 
  uint32_t sample = passed[sample_idx];
  mark_threads[sample_idx] = sample;
  calculate_data_point[sample_idx] = sample;

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  float second_min_dist = FLT_MAX;
  uint32_t doffset = d_clusters_size * d_features_size;
  int16_t index = 0; 

  
    /* 4 */
    //-----------------------------------------------------------------------
  for(uint32_t c = 0 ; c < d_clusters_size ; c++){   
    
    if(c == cluster) {
      continue;
    }

    uint32_t group = groups[c];
    if (group >= d_yy_groups_size) {
      continue;
    }
    float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
    /* Local Filtering #1 */
    if (lower_bound >= upper_bound) {
      if (lower_bound < second_min_dist) {
        second_min_dist = lower_bound;
      }
      continue;
    }

    lower_bound += drifts[group] - drifts[doffset + c];
    /* Local Filtering #2 */
    if (second_min_dist < lower_bound) {
      continue;
    }
   
    mark_threads[(1 + index) * d_passed_number + sample_idx] = (c); //column-major & push
    index++;
    
    atomicAdd(&calculate_centroid[c], 1); // Per 1 centroid # of calculation with data point
    // atomicAdd(&calculate_data_point[d_passed_number + sample_idx], 1); // Per 1 data point # of calculation with centroid
    
  }
  __syncthreads();
  calculate_data_point[d_passed_number + sample_idx] = index;
  second_min_dists[sample] = second_min_dist; 
 }


//-----------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter_threshold(
    uint32_t *__restrict__ calculate_centroid, float *d_skip_threshold, uint32_t *d_partition_threshold){
 
  //printf("calculate_centroid[%u] = %u\n", 0, calculate_centroid[0]);
  uint32_t sum = 0;
  for(uint32_t i = 0 ; i < d_clusters_size ; i++){
    sum += calculate_centroid[i]; // correct
  }
  //printf("sum = %u\n",sum);
  /* Claculate Skip Ratio */
  float skip_threshold = 1.0f - (static_cast<float>(sum) / (d_passed_number * d_clusters_size)); // correct
  *d_skip_threshold = skip_threshold; // correct
  //printf("Skip Ratio = %.4f\n", skip_threshold);

  //*d_partition_threshold = static_cast<uint32_t>(static_cast<float>(d_passed_number) / static_cast<float>(15));
  //printf("partition threshold = %u\n", *d_partition_threshold);
  
}

//-----------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter_partition(
    const uint32_t *__restrict__ passed, uint32_t *__restrict__ calculate_data_point,
    uint32_t *__restrict__ mark_cpu, uint32_t *__restrict__ mark_gpu,
    float *d_skip_threshold,
    const uint32_t *d_partition_threshold, uint32_t *d_sum1, uint32_t *d_sum2){
    // int *index_cpu, int *index_gpu) {

    uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (sample_idx >= d_passed_number) {
        return;  
    }

    uint32_t sample = passed[sample_idx]; 
    // if(*d_skip_threshold < 0.2){    // Skip little
    //   if(sample_idx < 70000){
    //       uint32_t idx = atomicAdd(&d_mark_cpu_number, 1);  
    //       mark_cpu[idx] = sample;
    //   }else{
    //       uint32_t idx = atomicAdd(&d_mark_gpu_number, 1);  
    //       mark_gpu[idx] = sample;
    //   }
    // } 
    // else{                           // Skip a lot
      // if(calculate_data_point[sample_idx] < 512){
      //     uint32_t idx = atomicAdd(&d_mark_cpu_number, 1);  
      //     mark_cpu[idx] = sample;
      // }else{
      //     uint32_t idx = atomicAdd(&d_mark_gpu_number, 1);  
      //     mark_gpu[idx] = sample;
      // }
    // }
    
    uint32_t threshold = static_cast<uint32_t>((static_cast<float>(d_passed_number) / static_cast<float>(16) * static_cast<float>(12)));
    if(sample < 70000){
      uint32_t idx = atomicAdd(&d_mark_cpu_number, 1);  
      mark_cpu[idx] = sample;
      *d_sum1 += calculate_data_point[sample];
    }else{
      uint32_t idx = atomicAdd(&d_mark_gpu_number, 1);  
      mark_gpu[idx] = sample;
      *d_sum2 += calculate_data_point[sample];
    }

    //mark_gpu[atomicAdd(&d_mark_gpu_number,1)] = sample;

}

//-----------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter2_1(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const uint32_t *__restrict__ passed, const F *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads, uint32_t *__restrict__ calculate_data_point, uint32_t *__restrict__ calculate_centroid){
 
  volatile uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(sample_idx >= d_passed_number){return;}
 
  uint32_t sample = passed[sample_idx];
  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];
 
  uint32_t valid_centroids = calculate_data_point[d_passed_number + sample_idx];
    /* 4 */
    //-----------------------------------------------------------------------
  for (uint16_t i = 0; i < valid_centroids ; i++) { 
      uint16_t c = mark_threads[(i+1) * d_passed_number + sample_idx];
     // atomicAdd(&d_col[sample], 1);
      float dist = METRIC<M,F>::distance_t(samples, centroids + (c) * d_features_size, d_samples_size, sample);
      //float dist = METRIC<M,F>::distance_t( samples, centroids + (c) * d_features_size, d_samples_size, sample);
     
      if(dist < min_dist){
        second_min_dist = min_dist;
        min_dist = dist;
        nearest = c;
      }
      else if(dist < second_min_dist){
        second_min_dist = dist;
      }
  }   
 
  /* 5 */
  //---------------------------------------------------------------------------------------------------------------
  uint32_t nearest_group = groups[nearest];
  uint32_t previous_group = groups[cluster];
  bounds[static_cast<uint64_t>(length) * (1 + nearest_group) + sample] = second_min_dist; // lower bound adjust

  if (nearest_group != previous_group) {
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + previous_group) + sample;
    float pb = bounds[gindex];
    if (pb > upper_bound) {
      bounds[gindex] = upper_bound;
    }
  }
  bounds[sample] = min_dist;  // upper bound adjust
  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}
//-----------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter2_2(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const uint32_t *__restrict__ mark_gpu, 
    //const uint32_t *__restrict__ passed, 
    const F *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= d_mark_gpu_number) {
   //if (sample >= d_passed_number){
    return;
  }


  sample = mark_gpu[sample];
  //sample = passed[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  //printf("cluster = %u\n", cluster);
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];
  uint32_t doffset = d_clusters_size * d_features_size;
  //printf("GPU sample = %u upper_bound = min_dist = %f cluster = %u second_min_dist = %f \n", sample, min_dist, cluster,second_min_dist);
  
  extern __shared__ float shared_memory[];
  F *volatile shared_centroids = reinterpret_cast<F*>(shared_memory);
  const uint32_t cstep = d_shmem_size / d_features_size;
  const uint32_t size_each = cstep / min(blockDim.x, d_passed_number - blockIdx.x * blockDim.x) + 1;

  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
    uint32_t coffset = gc * d_features_size;
    __syncthreads();
    for (uint32_t i = 0; i < size_each; i++) {
      uint32_t ci = threadIdx.x * size_each + i;
      uint32_t local_offset = ci * d_features_size;
      uint32_t global_offset = coffset + local_offset;
      if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
        #pragma unroll 4
        for (int f = 0; f < d_features_size; f++) {
          shared_centroids[local_offset + f] = centroids[global_offset + f];
        }
      }
    }
    __syncthreads();

    for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
    
      if (c == cluster) {
        continue;
      }
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) { // 102
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      
      float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
      /* Local Filtering #1 */
      if (lower_bound >= upper_bound) {
        if (lower_bound < second_min_dist) {
          second_min_dist = lower_bound;
        }
       
        continue;
      }
      lower_bound += drifts[group] - drifts[doffset + c];
     
      /* Local Filtering #2 */
      if (second_min_dist < lower_bound) {
       
        continue;
      }
      float dist = METRIC<M, F>::distance_t(samples, shared_centroids + (c-gc) * d_features_size, d_samples_size, sample + offset);
    
      if (dist < min_dist) {
        second_min_dist = min_dist;
        min_dist = dist;
        nearest = c;
      } else if (dist < second_min_dist) {
        second_min_dist = dist;
      }
    }
  }


  uint32_t nearest_group = groups[nearest];
  uint32_t previous_group = groups[cluster];
  bounds[static_cast<uint64_t>(length) * (1 + nearest_group) + sample] = second_min_dist;
  if (nearest_group != previous_group) {
    uint64_t gindex = static_cast<uint64_t>(length) * (1 + previous_group) + sample;
    float pb = bounds[gindex];
    if (pb > upper_bound) {
      bounds[gindex] = upper_bound;
    }
  }
  bounds[sample] = min_dist;

  if (cluster != nearest) {
    assignments[sample] = nearest;
    atomicAggInc(&d_changed_number);
  }
}





//--------------------------------------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_calc_average_distance(
    uint32_t offset, uint32_t length, const F *__restrict__ samples,
    const F *__restrict__ centroids, const uint32_t *__restrict__ assignments,
    atomic_float *distance) {
  volatile uint64_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  float dist = 0;
  if (sample < length) {
    sample += offset;
    dist = METRIC<M, F>::distance_t(samples, centroids + assignments[sample] * d_features_size, d_samples_size, sample);
  }
  float sum = warpReduceSum(dist);
  if (threadIdx.x % 32 == 0) {
    atomicAdd(distance, sum);
  }
}

////////////////////------------------------------------------------------------
// Host functions //------------------------------------------------------------
////////////////////------------------------------------------------------------

static int check_changed(int iter, float tolerance, uint32_t h_samples_size, const std::vector<int> &devs, int32_t verbosity) {
  uint32_t overall_changed = 0;
  //uint32_t overall_changed = h_changed_number_cpu;
  FOR_EACH_DEV(
    uint32_t my_changed = 0;
    CUCH(cudaMemcpyFromSymbol(&my_changed, d_changed_number, sizeof(my_changed)), kmcudaMemoryCopyError);  // device -> host
    overall_changed += my_changed;
  );
  INFO("Iteration %d: %" PRIu32 " Reassignments\n", iter, overall_changed);
  //printf("\n");
  if (overall_changed <= tolerance * h_samples_size) {   // 0.009 * 1000000 = 9000
    return -1;
  }
  assert(overall_changed <= h_samples_size);
  uint32_t zero = 0;
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbolAsync(d_changed_number, &zero, sizeof(zero)), kmcudaMemoryCopyError);  // host -> device
  );
  return kmcudaSuccess;
}

static KMCUDAResult prepare_mem(
    uint32_t h_samples_size, uint32_t h_clusters_size, bool resume,
    const std::vector<int> &devs, int verbosity, udevptrs<uint32_t> *ccounts,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_prev,
    std::vector<uint32_t> *shmem_sizes) {
  uint32_t zero = 0;
  shmem_sizes->clear();
  FOR_EACH_DEVI(
    uint32_t h_shmem_size;
    CUCH(cudaMemcpyFromSymbol(&h_shmem_size, d_shmem_size, sizeof(h_shmem_size)), kmcudaMemoryCopyError);  // device -> host
    shmem_sizes->push_back(h_shmem_size * sizeof(uint32_t));  // add new element in end of vector
    CUCH(cudaMemcpyToSymbolAsync(d_changed_number, &zero, sizeof(zero)), kmcudaMemoryCopyError);
    if (!resume) {
      CUCH(cudaMemsetAsync((*ccounts)[devi].get(), 0, h_clusters_size * sizeof(uint32_t)), kmcudaRuntimeError);
      CUCH(cudaMemsetAsync((*assignments)[devi].get(), 0xff, h_samples_size * sizeof(uint32_t)), kmcudaRuntimeError);
      CUCH(cudaMemsetAsync((*assignments_prev)[devi].get(), 0xff, h_samples_size * sizeof(uint32_t)), kmcudaRuntimeError);
    }
  );
  return kmcudaSuccess;
}



extern "C" {


KMCUDAResult kmeans_cuda_setup(
    uint32_t h_samples_size, uint16_t h_features_size, uint32_t h_clusters_size,
    uint32_t h_yy_groups_size, const std::vector<int> &devs, int32_t verbosity) {

  FOR_EACH_DEV(

    printf("\n");

    //DEBUG("-------------------------------------------------------------------------------------------------------------\n");
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)), kmcudaMemoryCopyError);
   // cudaHostAlloc((void**)&samples, h_samples_size, cudaHostAllocMapped);
    //DEBUG("GPU #%" PRIu32 " has %d samples_size\n", dev, h_samples_size);
    CUCH(cudaMemcpyToSymbol(d_features_size, &h_features_size, sizeof(h_features_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " has %d features_size\n", dev, h_features_size);
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " has %d clusters_size\n", dev, h_clusters_size);
    CUCH(cudaMemcpyToSymbol(d_yy_groups_size, &h_yy_groups_size, sizeof(h_yy_groups_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " has %d yy_groups_size\n", dev, h_yy_groups_size);

    cudaDeviceProp props;
    CUCH(cudaGetDeviceProperties(&props, dev), kmcudaRuntimeError);
    int h_shmem_size = static_cast<int>(props.sharedMemPerBlock);
    //DEBUG("GPU #%" PRIu32 " has %d bytes of shared memory per block\n", dev, h_shmem_size); // h_shmem_size 49152
    h_shmem_size /= sizeof(uint32_t);                                                       // transfer unit h_shmem_size = 49152 / 4 = 12288
    CUCH(cudaMemcpyToSymbol(d_shmem_size, &h_shmem_size, sizeof(h_shmem_size)), kmcudaMemoryCopyError);
    //DEBUG("GPU #%" PRIu32 " translated shared memory per block is %d bytes\n", dev, h_shmem_size);
    //DEBUG("-------------------------------------------------------------------------------------------------------------\n");

    printf("\n");

  );

  return kmcudaSuccess;

}

KMCUDAResult kmeans_cuda_plus_plus(
    uint32_t h_samples_size, uint32_t h_features_size, uint32_t cc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<float> *dists, float *host_dists, atomic_float *dist_sum) {
  auto plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  uint32_t max_len = 0;
  for (auto &p : plan) {
    auto len = std::get<1>(p);
    if (max_len < len) {
      max_len = len;
    }
  }
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_plus_plus, <<<grid, block>>>(
        offset, length, cc,
        reinterpret_cast<const F*>(samples[devi].get()),
        reinterpret_cast<const F*>((*centroids)[devi].get()),
        (*dists)[devi].get(), dev_dists[devi].get()));
  );
  uint32_t dist_offset = 0;
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    dim3 block(BS_KMPP, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    CUCH(cudaMemcpyAsync(
        host_dists + offset, (*dists)[devi].get(),
        length * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
    dist_offset += grid.x;
  );
  atomic_float sum = 0;
  FOR_EACH_DEVI(
    if (std::get<1>(plan[devi]) == 0) {
      continue;
    }
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    sum += hdist;
  );
  *dist_sum = sum;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_calc_q(
    uint32_t h_samples_size, uint32_t h_features_size, uint32_t firstc,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int verbosity, const udevptrs<float> &samples, udevptrs<float> *d_q,
    float *h_q) {
  auto plan = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BS_AFKMC2_Q, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    int shmem = std::max(
        BS_AFKMC2_Q, static_cast<int>(h_features_size)) * sizeof(float);
    KERNEL_SWITCH(kmeans_afkmc2_calc_q_dists,
                  <<<grid, block, shmem>>>(
        offset, length, firstc,
        reinterpret_cast<const F*>(samples[devi].get()),
        (*d_q)[devi].get(), dev_dists[devi].get()));

  );
  atomic_float dists_sum = 0;
  FOR_EACH_DEVI(
    if (std::get<1>(plan[devi]) == 0) {
      continue;
    }
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    dists_sum += hdist;
  );
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    kmeans_afkmc2_calc_q<<<grid, block>>>(
        offset, length, dists_sum, (*d_q)[devi].get());
  );
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plan[devi];
    CUCH(cudaMemcpyAsync(h_q + offset, (*d_q)[devi].get() + offset, length * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
    FOR_OTHER_DEVS(
      CUP2P(d_q, offset, length);
    );
  );
  SYNC_ALL_DEVS;
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_random_step(
    uint32_t k, uint32_t m, uint64_t seed, int verbosity, const float *q,
    uint32_t *d_choices, uint32_t *h_choices, float *d_samples, float *h_samples) {
  dim3 block(BS_AFKMC2_R, 1, 1);
  dim3 grid(upper(m, block.x), 1, 1);
  kmeans_afkmc2_random_step<<<grid, block>>>(m, seed, k, q, d_choices, d_samples);
  CUCH(cudaMemcpy(h_choices, d_choices, m * sizeof(uint32_t), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  CUCH(cudaMemcpy(h_samples, d_samples, m * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_afkmc2_min_dist(
    uint32_t k, uint32_t m, KMCUDADistanceMetric metric, int fp16x2,
    int32_t verbosity, const float *samples, const uint32_t *choices,
    const float *centroids, float *d_min_dists, float *h_min_dists) {
  if (m > k || m > SHMEM_AFKMC2_MT) {
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(m, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_afkmc2_min_dist, <<<grid, block>>>(
        m, k, reinterpret_cast<const F*>(samples), choices,
        reinterpret_cast<const F*>(centroids), d_min_dists));
  } else {
    dim3 block(BS_AFKMC2_MDT, 1, 1);
    dim3 grid(upper(k, block.x), 1, 1);
    CUCH(cudaMemsetAsync(d_min_dists, 0xff, m * sizeof(float)), kmcudaRuntimeError);
    KERNEL_SWITCH(kmeans_afkmc2_min_dist_transposed,
        <<<grid, block, m * sizeof(float)>>>(
        m, k, reinterpret_cast<const F*>(samples), choices,
        reinterpret_cast<const F*>(centroids), d_min_dists));
  }
  CUCH(cudaMemcpy(h_min_dists, d_min_dists, m * sizeof(float), cudaMemcpyDeviceToHost), kmcudaMemoryCopyError);
  return kmcudaSuccess;
}

KMCUDAResult kmeans_cuda_lloyd(
    float tolerance, uint32_t h_samples_size, uint32_t h_clusters_size,
    uint16_t h_features_size, KMCUDADistanceMetric metric, bool resume,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, int *iterations = nullptr) {
  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, resume, devs, verbosity,
                     ccounts, assignments, assignments_prev, &shmem_sizes));
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  auto planc = distribute(h_clusters_size, h_features_size * sizeof(float), devs);
  if (verbosity > 1) {
    // print_plan("plans", plans);
    // print_plan("planc", planc);
  }
  dim3 sblock(BS_LL_ASS, 1, 1);
  dim3 cblock(BS_LL_CNT, 1, 1);
  for (int iter = 1; ; iter++) {
    if (!resume || iter > 1) {
      FOR_EACH_DEVI(
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
        if (length == 0) {
          continue;
        }
        dim3 sgrid(upper(length, sblock.x), 1, 1);
        int shmem_size = shmem_sizes[devi];
        int64_t ssqrmem = sblock.x * h_features_size * sizeof(float);
        if (shmem_size > ssqrmem && shmem_size - ssqrmem >=
            static_cast<int>((h_features_size + 1) * sizeof(float))) {
          KERNEL_SWITCH(kmeans_assign_lloyd_smallc, <<<sgrid, sblock, shmem_size>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments_prev)[devi].get() + offset,
              (*assignments)[devi].get() + offset));
        } else {
          KERNEL_SWITCH(kmeans_assign_lloyd, <<<sgrid, sblock, shmem_size>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments_prev)[devi].get() + offset,
              (*assignments)[devi].get() + offset));
        }
      );
      FOR_EACH_DEVI(
        uint32_t offset, length;
        std::tie(offset, length) = plans[devi];
        if (length == 0) {
          continue;
        }
        FOR_OTHER_DEVS(
          CUP2P(assignments_prev, offset, length);
          CUP2P(assignments, offset, length);
        );
      );
      int status = check_changed(iter, tolerance, h_samples_size, devs, verbosity);
      //printf("Check Change\n"); // Check for lloyd's K-Means
      if (status < kmcudaSuccess) {
        if (iterations) {
          *iterations = iter;
        }
        return kmcudaSuccess;
      }
      if (status != kmcudaSuccess) {
        return static_cast<KMCUDAResult>(status);
      }
    }
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_adjust, <<<cgrid, cblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get(),
          reinterpret_cast<F*>((*centroids)[devi].get()), (*ccounts)[devi].get()));
    );
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
  }
}

KMCUDAResult kmeans_cuda_yy(
    float tolerance, uint32_t h_yy_groups_size, uint32_t h_samples_size, uint32_t h_clusters_size, uint16_t h_features_size,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids, udevptrs<uint32_t> *ccounts, 
    udevptrs<uint32_t> *assignments_prev, udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_yy,
    udevptrs<float> *centroids_yy, udevptrs<float> *bounds_yy, udevptrs<float> *drifts_yy, udevptrs<uint32_t> *passed_yy, 
    udevptrs<float> *second_min_dists_yy) {
  if (h_yy_groups_size == 0 || YINYANG_DRAFT_REASSIGNMENTS <= tolerance) {
    if (verbosity > 0) {
      if (h_yy_groups_size == 0) {
        printf("too few clusters for this yinyang_t => Lloyd\n");
      } else {
        printf("tolerance is too high (>= %.2f) => Lloyd\n", YINYANG_DRAFT_REASSIGNMENTS);
      }
    }
    return kmeans_cuda_lloyd(
        tolerance, h_samples_size, h_clusters_size, h_features_size, metric,
        false, devs, fp16x2, verbosity, samples, centroids, ccounts,
        assignments_prev, assignments);
  }
  INFO("running Lloyd until reassignments drop below %" PRIu32 "\n", (uint32_t)(YINYANG_DRAFT_REASSIGNMENTS * h_samples_size));
  int iter;
  RETERR(kmeans_cuda_lloyd(
      YINYANG_DRAFT_REASSIGNMENTS, h_samples_size, h_clusters_size,
      h_features_size, metric, false, devs, fp16x2, verbosity, samples,
      centroids, ccounts, assignments_prev, assignments, &iter));
  if (check_changed(iter, tolerance, h_samples_size, devs, 0) < kmcudaSuccess) {
    return kmcudaSuccess;
  }
  // map each centroid to yinyang group -> assignments_yy
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_clusters_size, sizeof(h_samples_size)), kmcudaMemoryCopyError);    // host -> device
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_yy_groups_size, sizeof(h_yy_groups_size)), kmcudaMemoryCopyError);    // host -> device
  );

  {
    udevptrs<float> tmpbufs, tmpbufs2;
    auto max_slength = max_distribute_length(h_samples_size, h_features_size * sizeof(float), devs);
    for (auto &pyy : *passed_yy) {
      // max_slength is guaranteed to be greater than or equal to
      // h_clusters_size + h_yy_groups_size  1024 + 102
      tmpbufs.emplace_back(reinterpret_cast<float*>(pyy.get()) +
          max_slength - h_clusters_size - h_yy_groups_size, true);
      tmpbufs2.emplace_back(tmpbufs.back().get() + h_clusters_size, true);
    }
    RETERR(cuda_transpose(
        h_clusters_size, h_features_size, true, devs, verbosity, centroids));
    RETERR(kmeans_init_centroids(
        kmcudaInitMethodPlusPlus, nullptr, h_clusters_size, h_features_size,
        h_yy_groups_size, metric, 0, devs, -1, fp16x2, verbosity, nullptr,
        *centroids, &tmpbufs, nullptr, centroids_yy),
           INFO("kmeans_init_centroids() failed for yinyang groups: %s\n", cudaGetErrorString(cudaGetLastError())));
    RETERR(kmeans_cuda_lloyd(
        YINYANG_GROUP_TOLERANCE, h_clusters_size, h_yy_groups_size, h_features_size,
        metric, false, devs, fp16x2, verbosity, *centroids, centroids_yy,
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs2),
        reinterpret_cast<udevptrs<uint32_t> *>(&tmpbufs), assignments_yy));
    RETERR(cuda_transpose(
        h_clusters_size, h_features_size, false, devs, verbosity, centroids));
  }
  /* h_samples_size, h_clusters_size host to device*/
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_samples_size, &h_samples_size, sizeof(h_samples_size)), kmcudaMemoryCopyError); // host -> device
    CUCH(cudaMemcpyToSymbol(d_clusters_size, &h_clusters_size, sizeof(h_clusters_size)), kmcudaMemoryCopyError); // host -> device
  );

  std::vector<uint32_t> shmem_sizes;
  RETERR(prepare_mem(h_samples_size, h_clusters_size, true, devs, verbosity, ccounts, assignments, assignments_prev, &shmem_sizes));
  dim3 siblock(BS_YY_INI, 1, 1);
  dim3 sgblock(BS_YY_GFL, 1, 1);
  dim3 slblock(BS_YY_LFL, 1, 1);
  dim3 cblock(BS_LL_CNT, 1, 1);
  dim3 gblock(BLOCK_SIZE, 1, 1);
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs); // sample data 
  auto planc = distribute(h_clusters_size, h_features_size * sizeof(float), devs); // centroid data
  auto plang = distribute(h_yy_groups_size, h_features_size * sizeof(float), devs); // centroid group data 

  if (verbosity > 1) {
    // print_plan("plans", plans);
    // print_plan("planc", planc);
    // print_plan("plang", plang);
  }

  bool refresh = true;
  uint32_t h_passed_number = 0; // initialization
  uint32_t h_mark_gpu_number = 0;
  uint32_t h_mark_cpu_number = 0;

  //------------------------------------------------------iter-------------------------------------------------------------
  for (; ; iter++) {
    if (!refresh) { // refresh: decine whether initialize bounds -> do not initialize
      int status = check_changed(iter-1, tolerance, h_samples_size, devs, verbosity);
      //printf("Check Changed Yinynag, Current Iteration = %d\n", iter); // check for Yinyang K-Means
      if (status < kmcudaSuccess) {
        return kmcudaSuccess;
      }
      if (status != kmcudaSuccess) {
        return static_cast<KMCUDAResult>(status);
      }

      FOR_EACH_DEV(
        uint32_t local_passed;
        CUCH(cudaMemcpyFromSymbol(&local_passed, d_passed_number, sizeof(h_passed_number)), kmcudaMemoryCopyError); // device -> host
        h_passed_number += local_passed;
        uint32_t local_mark_gpu_number;
        uint32_t local_mark_cpu_number;
        CUCH(cudaMemcpyFromSymbol(&local_mark_gpu_number, d_mark_gpu_number, sizeof(h_mark_gpu_number)), kmcudaMemoryCopyError);
        h_mark_gpu_number += local_mark_gpu_number;
        CUCH(cudaMemcpyFromSymbol(&local_mark_cpu_number, d_mark_cpu_number, sizeof(h_mark_cpu_number)), kmcudaMemoryCopyError);
        h_mark_cpu_number += local_mark_cpu_number;
      );
      //DEBUG("Iteration %u passed number: %" PRIu32 "\n", iter-1, h_passed_number);
      if (1.f - (h_passed_number + 0.f) / h_samples_size < YINYANG_REFRESH_EPSILON) {
        refresh = true;
      }
      h_passed_number = 0;
      h_mark_gpu_number = 0;
      h_mark_cpu_number = 0;
      //printf("Executed!"); correct
    }
//------------------------------------------------------------------------------------yinyang 1---------------------------------------------------------------------------
    if (refresh) {    // initialize
      printf("\n");
      INFO("Refreshing Yinyang Bounds, Call Bound Initialization Kernel\n");
      printf("\n");
      FOR_EACH_DEVI(
        uint32_t offset, length; // offset: start of data, length: size of data 
        std::tie(offset, length) = plans[devi];
        if (length == 0) {  // if there is no daa that device will process, 
          continue;
        }


        /* Initialize the host timer */
        cudaEvent_t start1, stop1;
        /* Create CUDA events */
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        /* Start to measure the execution time */
        cudaEventRecord(start1);      

        dim3 sigrid(upper(length, siblock.x), 1, 1);
        KERNEL_SWITCH(kmeans_yy_init, <<<sigrid, siblock, shmem_sizes[devi]>>>(
            offset, length,
            reinterpret_cast<const F*>(samples[devi].get()),
            reinterpret_cast<const F*>((*centroids)[devi].get()),
            (*assignments)[devi].get() + offset,
            (*assignments_yy)[devi].get(), (*bounds_yy)[devi].get()));
        cudaDeviceSynchronize();
        /* Record the event right after the CUDA kernel execution finished */
        cudaEventRecord(stop1);
        /* Synchronize the device to measure the execution time from the host side */
        cudaEventSynchronize(stop1);
        /* Calculate and print the elapsed time */
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start1, stop1);
        printf("(1) Initialization kernel execution time: %f ms\n", milliseconds1);
        /* Destroy the CUDA events */
        cudaEventDestroy(start1);
        cudaEventDestroy(stop1); 
        
      );

      refresh = false;
    }
//----------------------------------------------------------------------------yinyang 2-----------------------------------------------------------------------------------

    CUMEMCPY_D2D_ASYNC(*drifts_yy, 0, *centroids, 0, h_clusters_size * h_features_size);  // asynchronously copy data from centroids to drifts
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];   // tuple(offset, length)
      if (length == 0) {
        continue;
      }
      
      /* Initialize the host timer */
      cudaEvent_t start2, stop2;
      /* Create CUDA events */
      cudaEventCreate(&start2);
      cudaEventCreate(&stop2);
      /* Start to measure the execution time */
      cudaEventRecord(start2);

      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_adjust, <<<cgrid, cblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get(),
          reinterpret_cast<F*>((*centroids)[devi].get()), (*ccounts)[devi].get()));
      cudaDeviceSynchronize();
      /* Record the event right after the CUDA kernel execution finished */
      cudaEventRecord(stop2);
      /* Synchronize the device to measure the execution time from the host side */
      cudaEventSynchronize(stop2);
      /* Calculate and print the elapsed time */
      float milliseconds2 = 0;
      cudaEventElapsedTime(&milliseconds2, start2, stop2);
      printf("\n");
      printf("[Current Iteration = %d]\n", iter);
      printf("(2) Centroid Adjust kernel execution time: %f ms\n", milliseconds2);

      /* Destroy the CUDA events */
      cudaEventDestroy(start2);
      cudaEventDestroy(stop2); 
      
    );
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS( // if need. copy to other device
        CUP2P(ccounts, offset, length);
        CUP2P(centroids, offset * h_features_size, length * h_features_size);
      );
    );
//----------------------------------------------------------------------------yinyang 3------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }

      /* Initialize the host timer */
      cudaEvent_t start3, stop3;
      /* Create CUDA events */
      cudaEventCreate(&start3);
      cudaEventCreate(&stop3);
      /* Start to measure the execution time */
      cudaEventRecord(start3);  

      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_calc_drifts, <<<cgrid, cblock>>>(
          offset, length, reinterpret_cast<const F*>((*centroids)[devi].get()),
          reinterpret_cast<F*>((*drifts_yy)[devi].get())));
      cudaDeviceSynchronize();
      /* Record the event right after the CUDA kernel execution finished */
      cudaEventRecord(stop3);
      /* Synchronize the device to measure the execution time from the host side */
      cudaEventSynchronize(stop3);
      /* Calculate and print the elapsed time */
      float milliseconds3 = 0;
      cudaEventElapsedTime(&milliseconds3, start3, stop3);
      printf("(3) Drift Calculation kernel execution time: %f ms\n", milliseconds3);
      /* Destroy the CUDA events */
      cudaEventDestroy(start3);
      cudaEventDestroy(stop3); 
      
    );
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = planc[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(drifts_yy, h_clusters_size * h_features_size + offset, length);
      );
    );
//---------------------------------------------------------------------------------yinyang 4--------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
      if (length == 0) {
        continue;
      }
      

      /* Initialize the host timer */
      cudaEvent_t start4, stop4;
      /* Create CUDA events */
      cudaEventCreate(&start4);
      cudaEventCreate(&stop4);
      /* Start to measure the execution time */
      cudaEventRecord(start4);
      dim3 ggrid(upper(length, gblock.x), 1, 1);
      kmeans_yy_find_group_max_drifts<<<ggrid, gblock, shmem_sizes[devi]>>>(
          offset, length, (*assignments_yy)[devi].get(),
          (*drifts_yy)[devi].get());
      cudaDeviceSynchronize();
      /* Record the event right after the CUDA kernel execution finished */
      cudaEventRecord(stop4);
      /* Synchronize the device to measure the execution time from the host side */
      cudaEventSynchronize(stop4);
      /* Calculate and print the elapsed time */
      float milliseconds4 = 0;
      cudaEventElapsedTime(&milliseconds4, start4, stop4);
      printf("(4) Group Max Drifts kernel execution time: %f ms\n", milliseconds4);

      /* Destroy the CUDA events */
      cudaEventDestroy(start4);
      cudaEventDestroy(stop4);  
      
    );
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plang[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(drifts_yy, offset, length);
      );
    );

    FOR_EACH_DEV(
      CUCH(cudaMemcpyToSymbolAsync(d_passed_number, &h_passed_number, sizeof(h_passed_number)), kmcudaMemoryCopyError);  // host -> device
      CUCH(cudaMemcpyToSymbolAsync(d_mark_gpu_number, &h_mark_gpu_number, sizeof(h_mark_gpu_number)), kmcudaMemoryCopyError);
      CUCH(cudaMemcpyToSymbolAsync(d_mark_cpu_number, &h_mark_cpu_number, sizeof(h_mark_cpu_number)), kmcudaMemoryCopyError);
      
    );
//--------------------------------------------------------------------------------yinyang 5 ~ 7-------------------------------------------------------------------------------
   
    FOR_EACH_DEVI(
      //uint32_t h_distance_number = 0; // initialization
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      if (length == 0) {
        continue;
      }
      
      /* Initialize the host timer */
      cudaEvent_t start5, stop5;
      /* Create CUDA events */
      cudaEventCreate(&start5);
      cudaEventCreate(&stop5);
      /* Start to measure the execution time */
      cudaEventRecord(start5);
      dim3 sggrid(upper(length, sgblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_global_filter, <<<sggrid, sgblock>>>(
          offset, length,
          reinterpret_cast<const F*>(samples[devi].get()),
          reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), (*drifts_yy)[devi].get(),
          (*assignments)[devi].get() + offset, (*assignments_prev)[devi].get() + offset,
          (*bounds_yy)[devi].get(), (*passed_yy)[devi].get()));
      cudaDeviceSynchronize();
      /* Record the event right after the CUDA kernel execution finished */
      cudaEventRecord(stop5);
      /* Synchronize the device to measure the execution time from the host side */
      cudaEventSynchronize(stop5);
      /* Calculate and print the elapsed time */
      float milliseconds5 = 0;
      cudaEventElapsedTime(&milliseconds5, start5, stop5);
      printf("(5) Global Filter kernel execution time: %f ms\n", milliseconds5);
      /* Destroy the CUDA events */
      cudaEventDestroy(start5);
      cudaEventDestroy(stop5);
      /* For Print Out After Global Filter */
      uint32_t global_to_local;
      CUCH(cudaMemcpyFromSymbol(&global_to_local, d_passed_number, sizeof(uint32_t)), kmcudaMemoryCopyError);
      printf("(5) Iteration %d Global Filter to Local Filter: %u\n", iter, global_to_local);

//------------------------------------------------------------------------------------------------------------------------------------------------------------      
      int32_t *d_mark_threads;
      cudaMalloc(&d_mark_threads, static_cast<int32_t>(global_to_local) * static_cast<int32_t>(h_clusters_size + 1) * sizeof(int32_t));
      uint32_t *d_calculate_data_point;
      cudaMalloc(&d_calculate_data_point, static_cast<uint32_t>(global_to_local * 2) * sizeof(uint32_t));
      uint32_t *d_calculate_centroid;
      cudaMalloc(&d_calculate_centroid, static_cast<uint32_t>(h_clusters_size) * sizeof(uint32_t));   

      cudaMemset(d_mark_threads, 0xff, static_cast<int32_t>(global_to_local) * static_cast<int32_t>(h_clusters_size + 1) * sizeof(int32_t));
      cudaMemset(d_calculate_data_point, 0, static_cast<uint32_t>(global_to_local * 2) * sizeof(uint32_t));
      cudaMemset(d_calculate_centroid, 0, static_cast<uint32_t>(h_clusters_size) * sizeof(uint32_t));
      cudaDeviceSynchronize();


      cudaEvent_t start6, stop6;
      cudaEventCreate(&start6);
      cudaEventCreate(&stop6);
      cudaEventRecord(start6);

      dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter1, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*passed_yy)[devi].get(), reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), (*drifts_yy)[devi].get(),    //assignments_yy equal to groups??
          (*assignments)[devi].get() + offset, (*bounds_yy)[devi].get(), (*second_min_dists_yy)[devi].get(),
          //(*mark_threads_yy)[devi].get(), (*calculate_data_point_yy)[devi].get(), (*calculate_centroid_yy)[devi].get()
          d_mark_threads, d_calculate_data_point, d_calculate_centroid));
      cudaDeviceSynchronize();

      
      cudaEventRecord(stop6);
      cudaEventSynchronize(stop6);
      float milliseconds6 = 0;
      cudaEventElapsedTime(&milliseconds6, start6, stop6);
      printf("(6) Local Filter Marking kernel execution time: %f ms\n", milliseconds6);
      cudaEventDestroy(start6);
      cudaEventDestroy(stop6); 

      // uint32_t *h_calculate_data_point = (uint32_t*)malloc(global_to_local * 2 * sizeof(uint32_t)); 
      // cudaMemcpy(h_calculate_data_point, d_calculate_data_point, global_to_local * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      // for (uint32_t i = global_to_local; i < global_to_local * 2; i++) {
      //     printf("iter = %d, data_point[%6u] = %u\n", iter, (i-global_to_local), h_calculate_data_point[i]);
      // }
//------------------------------------------------------------------------------------------------------------------------------------------------------------

      
      float *h_skip_threshold;
      cudaHostAlloc((void**)&h_skip_threshold, sizeof(float), cudaHostAllocMapped);
      float *d_skip_threshold;
      cudaHostGetDevicePointer((void**)&d_skip_threshold, h_skip_threshold, 0);
      *h_skip_threshold = 0.0f;

      uint32_t *h_partition_threshold;
      cudaHostAlloc((void**)&h_partition_threshold, sizeof(uint32_t), cudaHostAllocMapped);
      uint32_t *d_partition_threshold;
      cudaHostGetDevicePointer((void**)&d_partition_threshold, h_partition_threshold, 0);
      *h_partition_threshold = 0;

      uint32_t *h_sum1;
      cudaHostAlloc((void**)&h_sum1, sizeof(uint32_t), cudaHostAllocMapped);
      uint32_t *d_sum1;
      cudaHostGetDevicePointer((void**)&d_sum1, h_sum1, 0);
      *h_sum1 = 0;

      uint32_t *h_sum2;
      cudaHostAlloc((void**)&h_sum2, sizeof(uint32_t), cudaHostAllocMapped);
      uint32_t *d_sum2;
      cudaHostGetDevicePointer((void**)&d_sum2, h_sum2, 0);
      *h_sum2 = 0;

      cudaEvent_t start7, stop7;
      cudaEventCreate(&start7);
      cudaEventCreate(&stop7);
      cudaEventRecord(start7);



      KERNEL_SWITCH(kmeans_yy_local_filter_threshold, <<<1, h_clusters_size>>>(
        d_calculate_centroid,
        d_skip_threshold, d_partition_threshold));
      cudaDeviceSynchronize();

      cudaEventRecord(stop7);
      cudaEventSynchronize(stop7);
      float milliseconds7 = 0;
      cudaEventElapsedTime(&milliseconds7, start7, stop7);
      printf("(7) Threshold Setting kernel execution time: %f ms\n", milliseconds7);
      cudaEventDestroy(start7);
      cudaEventDestroy(stop7); 
      
      
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

      cudaEvent_t start8, stop8;
      cudaEventCreate(&start8);
      cudaEventCreate(&stop8);
      cudaEventRecord(start8);

      uint32_t *d_mark_cpu;
      cudaMalloc(&d_mark_cpu, static_cast<uint32_t>(global_to_local) * sizeof(uint32_t));
      uint32_t *d_mark_gpu;
      cudaMalloc(&d_mark_gpu, static_cast<uint32_t>(global_to_local) * sizeof(uint32_t));

      cudaMemset(d_mark_cpu, 0, static_cast<uint32_t>(global_to_local) * sizeof(int32_t));
      cudaMemset(d_mark_gpu, 0, static_cast<uint32_t>(global_to_local) * sizeof(int32_t));    
      cudaDeviceSynchronize();

      KERNEL_SWITCH(kmeans_yy_local_filter_partition, <<<slgrid, slblock>>>((*passed_yy)[devi].get(), d_calculate_data_point, 
                                                                            d_mark_cpu, d_mark_gpu, d_skip_threshold, d_partition_threshold, d_sum1, d_sum2));
      cudaDeviceSynchronize();

      cudaEventRecord(stop8);
      cudaEventSynchronize(stop8);
      float milliseconds8 = 0;
      cudaEventElapsedTime(&milliseconds8, start8, stop8);
      printf("(8) CPU-GPU Data Point Partition kernel execution time: %f ms\n", milliseconds8);
      cudaEventDestroy(start8);
      cudaEventDestroy(stop8); 

    //  CUCH(cudaMemcpyFromSymbol(&h_mark_cpu_number, d_mark_cpu_number, sizeof(h_mark_cpu_number)), kmcudaMemoryCopyError);
    //  CUCH(cudaMemcpyFromSymbol(&h_mark_gpu_number, d_mark_gpu_number, sizeof(h_mark_gpu_number)), kmcudaMemoryCopyError);
      
    //  printf("CPU Threads = %u GPU Threads = %u Thread ratio = %.2f\n", h_mark_cpu_number, h_mark_gpu_number, (static_cast<float>(h_mark_gpu_number)/static_cast<float>(h_mark_cpu_number)));
      printf("Skip Ratio = %.4f\n", *h_skip_threshold);
      printf("GPU # of calculation = %6u\n", *h_sum1);
      printf("CPU # of calculation = %6u\n", *h_sum2);
//---------------------------------------------------------------------------------------------------------------------------------------------------
  
  if (*h_skip_threshold >= 0.9){         // solution
     
      cudaEvent_t start9, stop9;
      cudaEventCreate(&start9);
      cudaEventCreate(&stop9);
      cudaEventRecord(start9);
      
      
      dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter2_1, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*passed_yy)[devi].get(), reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), (*drifts_yy)[devi].get(),    //assignments_yy == groups??
          (*assignments)[devi].get() + offset, (*bounds_yy)[devi].get(),
          (*second_min_dists_yy)[devi].get(), 
          d_mark_threads, d_calculate_data_point, d_calculate_centroid));
      cudaDeviceSynchronize();

      cudaEventRecord(stop9);
      cudaEventSynchronize(stop9);
      float milliseconds9 = 0;
      cudaEventElapsedTime(&milliseconds9, start9, stop9);
      printf("(9-1) (GPU) Warp Divergence Resolve kernel execution time: %f ms\n", milliseconds9);
      cudaEventDestroy(start9);
      cudaEventDestroy(stop9); 
 
  }else{    /* start heterogeneous computing */

//----------------------------------------------------------------------------------------------------------------------------------------------
      cudaEvent_t start_d2h, stop_d2h;
      cudaEventCreate(&start_d2h);
      cudaEventCreate(&stop_d2h);
      cudaEventRecord(start_d2h);
      /* Host Malloc */
      uint32_t  *h_mark_cpu       = (uint32_t*)malloc(global_to_local * sizeof(uint32_t)); 
      float     *h_samples        = (float*)malloc(h_samples_size * h_features_size * sizeof(float));      
      float     *h_centroids      = (float*)malloc(h_clusters_size * h_features_size * sizeof(float));
      uint32_t  *h_assignments_yy = (uint32_t*)malloc(h_clusters_size * sizeof(uint32_t));
      float     *h_drifts_yy      = (float*)malloc((h_clusters_size) * (h_features_size + 1) * sizeof(float));
      uint32_t  *h_assignments    = (uint32_t*)malloc(h_samples_size * sizeof(uint32_t)); //problem
      float     *h_bounds_yy      = (float*)malloc(h_samples_size * (h_yy_groups_size + 1) * sizeof(float));
      float     *h_second_min_dists_yy = (float*)malloc(h_samples_size * sizeof(float));
      // uint32_t  *calculate_centroid_cpu = (uint32_t*)malloc(h_clusters_size * sizeof(uint32_t));
      // memset(calculate_centroid_cpu, 0, h_clusters_size * sizeof(uint32_t));
      
      /* D2H */
      cudaMemcpy(h_mark_cpu, d_mark_cpu, global_to_local * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_samples, (samples)[devi].get(), h_samples_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_centroids, (*centroids)[devi].get(), h_clusters_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_assignments_yy, (*assignments_yy)[devi].get(), h_clusters_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_drifts_yy, (*drifts_yy)[devi].get(), (h_clusters_size) * (h_features_size + 1) * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_assignments, (*assignments)[devi].get(), h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_bounds_yy, (*bounds_yy)[devi].get(), h_samples_size * (h_yy_groups_size + 1) * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_second_min_dists_yy, (*second_min_dists_yy)[devi].get(), h_samples_size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      cudaEventRecord(stop_d2h);
      cudaEventSynchronize(stop_d2h);
      float milliseconds_d2h = 0;
      cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h);
      printf("(9-2-1) Device to Host Data Transfer Time: %f ms\n", milliseconds_d2h);
      cudaEventDestroy(start_d2h);
      cudaEventDestroy(stop_d2h);
//----------------------------------------------------------------------------------------------------------------------------------------------------------------        
      /* GPU Kernel */
      cudaEvent_t start10, stop10;
      cudaEventCreate(&start10);
      cudaEventCreate(&stop10);
      cudaEventRecord(start10);
      
      //CUCH(cudaMemcpyFromSymbol(&h_mark_gpu_number, d_mark_gpu_number, sizeof(h_mark_gpu_number)), kmcudaMemoryCopyError); // device -> host
      //dim3 slgrid = dim3((h_mark_gpu_number + slblock.x - 1) / slblock.x, 1, 1);
      //printf("# of Thread Blocks = %d h_mark_gpu_number = %u \n", slgrid.x, h_mark_gpu_number);
      KERNEL_SWITCH(kmeans_yy_local_filter2_2, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          d_mark_gpu, reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), (*drifts_yy)[devi].get(),    //assignments_yy equal to groups??
          (*assignments)[devi].get() + offset, (*bounds_yy)[devi].get(),
          (*second_min_dists_yy)[devi].get()));
      cudaEventRecord(stop10);
      cudaEventSynchronize(stop10);
      float milliseconds10 = 0;
      cudaEventElapsedTime(&milliseconds10, start10, stop10);
      printf("(9-2-2) (GPU) GPU Local Filter execution time: %f ms\n", milliseconds10);
      cudaEventDestroy(start10);
      cudaEventDestroy(stop10); 

      /* CPU Function */
      uint32_t h_changed_number_cpu = 0;
      auto start_cpu = std::chrono::high_resolution_clock::now(); // CPU start time
      local_filter_cpu<float>(h_samples_size, h_clusters_size, h_features_size, h_yy_groups_size, h_samples, h_mark_cpu, h_centroids, h_assignments_yy, h_drifts_yy, h_assignments, h_bounds_yy, h_second_min_dists_yy, h_mark_cpu_number, h_changed_number_cpu);
      auto end_cpu = std::chrono::high_resolution_clock::now(); // CPU end time

      std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu; // CPU execution time
      std::cout << "(9-2-2) (CPU) CPU Local Filter execution time: " << cpu_duration.count() << " ms" << std::endl;
      //std::cout << "(9-2-2) (CPU) Changed Number: " << h_changed_number_cpu << std::endl; // Output the number of changes

      cudaDeviceSynchronize();     

      // uint32_t sum = 0;
      // for(uint32_t i = 0 ; i < h_clusters_size; i++){
      //   printf("calculate_centroid_cpu[%u] = %u \n", i, calculate_centroid_cpu[i]);
      //   sum += calculate_centroid_cpu[i];
      // }
      // printf("CPU_Total_calculation = %u\n", sum);
//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      
      /* H2D */     
      cudaEvent_t start_h2d, stop_h2d;
      cudaEventCreate(&start_h2d);
      cudaEventCreate(&stop_h2d);
      cudaEventRecord(start_h2d);
      
      for(uint32_t i = 0 ; i < h_mark_cpu_number ; i++){
        uint32_t sample = h_mark_cpu[i];
        cudaMemcpy((*bounds_yy)[devi].get() + sample * (h_yy_groups_size + 1), h_bounds_yy + sample * (h_yy_groups_size + 1), (h_yy_groups_size + 1) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((*assignments)[devi].get() + sample, h_assignments + sample, sizeof(uint32_t), cudaMemcpyHostToDevice);
      }

      cudaDeviceSynchronize();
      cudaEventRecord(stop_h2d);
      cudaEventSynchronize(stop_h2d);
      float milliseconds_h2d = 0;
      cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);
      printf("(9-2-3) Host To Device Data Transfer Time: %f ms\n", milliseconds_h2d);
      cudaEventDestroy(start_h2d);
      cudaEventDestroy(stop_h2d);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      /* Retrieve d_changed_number from GPU */
      uint32_t h_changed_number_gpu;
      CUCH(cudaMemcpyFromSymbol(&h_changed_number_gpu, d_changed_number, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost), kmcudaMemoryCopyError); // device -> host
      //printf("(9-2-4) (GPU) Changed Number: %u\n", h_changed_number_gpu);
      /* Combine CPU and GPU changed numbers */
      uint32_t total_changed_number = h_changed_number_cpu + h_changed_number_gpu;
      printf("(9-2-4) Iteration %d Total Reassignments (CPU + GPU): %u\n", iter, total_changed_number );
      CUCH(cudaMemcpyToSymbol(d_changed_number, &total_changed_number, sizeof(uint32_t), 0, cudaMemcpyHostToDevice), kmcudaMemoryCopyError); // reflect next iteration

//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      
      /* Memory Deallocation in Host */
      free(h_mark_cpu);
      free(h_assignments_yy);
      free(h_drifts_yy);
      free(h_assignments);
      free(h_bounds_yy);
      free(h_second_min_dists_yy);
      //free(calculate_centroid_cpu);


  }

  cudaFreeHost(h_skip_threshold);
  cudaFreeHost(h_sum1);
  cudaFreeHost(h_sum2);
  cudaFree(d_mark_threads);
  cudaFree(d_calculate_data_point);
  cudaFree(d_calculate_centroid);
  cudaFree(d_mark_cpu);
  cudaFree(d_mark_gpu);


    );
    
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FOR_EACH_DEVI(
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      if (length == 0) {
        continue;
      }
      FOR_OTHER_DEVS(
        CUP2P(assignments_prev, offset, length);
        CUP2P(assignments, offset, length);
      );

    );
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  }
  //------------------------------------------------------iter-------------------------------------------------------------
}




//-------------------------------------------------------------------------------------------------

KMCUDAResult kmeans_cuda_calc_average_distance(
    uint32_t h_samples_size, uint16_t h_features_size,
    KMCUDADistanceMetric metric, const std::vector<int> &devs, int fp16x2,
    int32_t verbosity, const udevptrs<float> &samples,
    const udevptrs<float> &centroids, const udevptrs<uint32_t> &assignments,
    float *average_distance) {
  INFO("calculating the average distance...\n");
  auto plans = distribute(h_samples_size, h_features_size * sizeof(float), devs);
  udevptrs<atomic_float> dev_dists;
  CUMALLOC(dev_dists, sizeof(atomic_float));
  CUMEMSET_ASYNC(dev_dists, 0, sizeof(atomic_float));
  FOR_EACH_DEVI(
    uint32_t offset, length;
    std::tie(offset, length) = plans[devi];
    if (length == 0) {
      continue;
    }
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(upper(length, block.x), 1, 1);
    KERNEL_SWITCH(kmeans_calc_average_distance,
                  <<<grid, block, block.x * sizeof(float)>>>(
        offset, length, reinterpret_cast<const F*>(samples[devi].get()),
        reinterpret_cast<const F*>(centroids[devi].get()),
        assignments[devi].get(), dev_dists[devi].get()));
  );
  atomic_float sum = 0;
  FOR_EACH_DEVI(
    atomic_float hdist;
    CUCH(cudaMemcpy(&hdist, dev_dists[devi].get(), sizeof(atomic_float),
                    cudaMemcpyDeviceToHost),
         kmcudaMemoryCopyError);
    sum += hdist;
  );
  *average_distance = sum / h_samples_size;
  return kmcudaSuccess;
}




}  // extern "C"