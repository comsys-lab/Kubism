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

#include "fp_abstraction.h"



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>



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


__constant__ uint32_t d_samples_size;
__constant__ uint32_t d_clusters_size;
__constant__ uint32_t d_yy_groups_size;
__constant__ int d_shmem_size;

float skip_threshold ;
float prev_execution_time;
float prev_skip_ratio;
uint32_t prev_global_to_local;

int8_t flag ; // reordering + warp divergence + heterogeneous computing or reordering + heterogeneous computing
int8_t cnt1 ; // kmeans_yy_init
int8_t cnt2 ; // skip theshold





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
    int32_t *__restrict__ mark_threads, uint16_t *__restrict__ calculate_data_point){
  
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample >= d_passed_number){ 
    return;
  }
 
  sample = passed[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  float second_min_dist = FLT_MAX;
  uint32_t doffset = d_clusters_size * d_features_size; 

  uint32_t bitmask_register = 0 ;

  for(uint32_t c = 0 ; c < d_clusters_size ; c++){   
    int mask_index = (c) / 32;  
    int bit_position = ( c) % 32;  

    if((c+1) % 32 == 0 || c == d_clusters_size - 1){
      mark_threads[sample * 32 + (mask_index)] = bitmask_register;
      bitmask_register = 0;
    }
    
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
    // int mask_index = (c) / 32;  
    // int bit_position = ( c) % 32;  

    bitmask_register |= (1 << bit_position);

    // if((c+1) % 32 == 0 || c == d_clusters_size - 1){
    //   mark_threads[sample * 32 + (mask_index)] = bitmask_register;
    //   bitmask_register = 0;
    // }
    //mark_threads[sample * 32 + mask_index] |= (1 << bit_position);

  second_min_dists[sample] = second_min_dist; 

 }
}
 //----------------------------------------------------------------------------------------------

 template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter1_1(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const uint32_t *__restrict__ passed, const F *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads, uint16_t *__restrict__ calculate_data_point, uint32_t *calculate_sum){
  
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if(sample >= d_passed_number){ 
    return;
  }
 
  sample = passed[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  float second_min_dist = FLT_MAX;
  uint32_t doffset = d_clusters_size * d_features_size;
  int16_t index = 0; 

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
    //mark_threads[index * d_samples_size + sample] = (c); //column-major & push
    index++;
    
  
  }
  
  //if(sample % 20 == 0){
    //calculate_data_point[sample] = index; // using in CPU
    atomicAdd(calculate_sum, index);
  //}

  second_min_dists[sample] = second_min_dist; 
  //atomicAdd(calculate_sum, index);
 }
 //-----------------------------------------------------------------------------------------------

template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter2_0_0(
    const uint32_t offset, const uint32_t length, const F *__restrict__ samples,
    const uint32_t *__restrict__ passed, const F *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds,
    float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads, uint16_t *__restrict__ calculate_data_point) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= d_passed_number){
    return;
  }

  sample = passed[sample];
  // if (calculate_data_point[sample] == 0) {
  //   return;
  // }

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
 //float second_min_dist = second_min_dists[sample];  
  float second_min_dist = FLT_MAX;
  uint32_t doffset = d_clusters_size * d_features_size;

  // Shared memory allocation
  // extern __shared__ float shared_centroids[];
  // const uint32_t cstep = d_shmem_size / d_features_size; // 클러스터별로 나누어 로드
  // const uint32_t size_each = cstep / min(blockDim.x, d_passed_number - blockIdx.x * blockDim.x) + 1;

//  for (uint32_t gc = 0; gc < d_clusters_size; gc += cstep) {
//     uint32_t coffset = gc * d_features_size;
//     __syncthreads();
//     for (uint32_t i = 0; i < size_each; i++) {
//       uint32_t ci = threadIdx.x * size_each + i;
//       uint32_t local_offset = ci * d_features_size;
//       uint32_t global_offset = coffset + local_offset;
//       if (global_offset < d_clusters_size * d_features_size && ci < cstep) {
//         #pragma unroll 4
//         for (int f = 0; f < d_features_size; f++) {
//           shared_centroids[local_offset + f] = centroids[global_offset + f];
//         }
//       }
//     }
//     __syncthreads();

    // Compute distances using shared memory
  // for (uint32_t c = gc; c < gc + cstep && c < d_clusters_size; c++) {
    for (uint32_t c = 0; c < d_clusters_size; c++) {
      if (c == cluster) {
        continue;
      }

      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }

      float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
      /* Local Filter Filtering 1 */
      if (lower_bound >= upper_bound) {
      //if(c >= 512){
        if (lower_bound < second_min_dist) {
          second_min_dist = lower_bound;
        }
        continue;
      }
      lower_bound += drifts[group] - drifts[doffset + c];
      /* Local Filter Filtering 2 */
      if (second_min_dist < lower_bound) {
      //if(c >= 512){
        continue;
      }

      // // Use shared memory for distance calculation
      // float dist = 0;
      // #pragma unroll 4
      // for (uint16_t f = 0; f < d_features_size; ++f) {
      //   float d = samples[d_samples_size * f + sample] - centroids[(c) * d_features_size + f];
      //   dist += d * d;
      // }
      // float dist1 = sqrt(dist);

      float dist = METRIC<M, F>::distance_t(
          samples, centroids + (c ) * d_features_size,
          d_samples_size, sample + offset);

      if (dist < min_dist) {
        second_min_dist = min_dist;
        min_dist = dist;
        nearest = c;
      } else if (dist < second_min_dist) {
        second_min_dist = dist;
      }
    }
 // }

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


template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter2_0_1(
    const uint32_t offset, const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ passed, const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds,
    float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads, uint16_t *__restrict__ calculate_data_point) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= d_passed_number){
    return;
  }

  sample = passed[sample];
  if (calculate_data_point[sample] == 0) {
    return;
  }

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];  
  
  uint32_t doffset = d_clusters_size * d_features_size;

  uint16_t num_centroids_to_process = calculate_data_point[sample];
  if(num_centroids_to_process == 0){
    return;
  }

  for (uint32_t c = 0 ; c < num_centroids_to_process ; c++) {
    
    if (c == cluster) {
      continue;
    }

    uint32_t group = groups[c];
    if (group >= d_yy_groups_size) {
      // this may happen if the centroid is insane (NaN)
      continue;
    }

    float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
    /* Local Filter Filtering 1 */
    if (lower_bound >= upper_bound) {
    //if(c >= 512){
      if (lower_bound < second_min_dist) {
        second_min_dist = lower_bound;
      }
      continue;
    }
    lower_bound += drifts[group] - drifts[doffset + c];
    /* Local Filter Filtering 2 */
    if (second_min_dist < lower_bound) {
    //if(c >= 512){
      continue;
    }

    // Use shared memory for distance calculation
    float dist = 0;
    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; ++f) {
      float d = samples[d_samples_size * f + sample] - centroids[(c) * d_features_size + f];
      dist += d * d;
    }
    float dist1 = sqrt(dist);

    if (dist1 < min_dist) {
      second_min_dist = min_dist;
      min_dist = dist1;
      nearest = c;
    } else if (dist1 < second_min_dist) {
      second_min_dist = dist1;
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



//-----------------------------------------------------------------------------------------------
/* 1. reordering */
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter2_1(    
    const uint32_t offset, const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ passed, const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, 
    //float *__restrict__ min_dists, uint32_t *__restrict__ nearests,
    float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads, uint16_t *__restrict__ calculate_data_point) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= d_passed_number){
    return;
  }

  sample = passed[sample];
  if (calculate_data_point[sample] == 0) {
    return;
  }

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];  
  
  uint32_t doffset = d_clusters_size * d_features_size;

  for (uint32_t c = 0 ; c < d_clusters_size; c++) {
      if (c == cluster) {
        continue;
      }
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      float lower_bound = bounds[
          static_cast<uint64_t>(length) * (1 + group) + sample];
      if (lower_bound >= upper_bound) {
        if (lower_bound < second_min_dist) {
          second_min_dist = lower_bound;
        }
        continue;
      }
      lower_bound += drifts[group] - drifts[doffset + c];
      if (second_min_dist < lower_bound) {
        continue;
      }
      // float dist = METRIC<M, F>::distance_t(samples, centroids + (c) * d_features_size, d_samples_size, sample + offset);

      float dist = 0;

      #pragma unroll 4
      for (uint16_t f = 0; f < d_features_size; ++f) {
        float d = samples[d_samples_size * f + sample] - centroids[c * d_features_size + f];

          dist += d * d;
      }
      float dist1 = sqrt(dist);

      if (dist1 < min_dist) {
        second_min_dist = min_dist;
        min_dist = dist1;
        nearest = c;
      } else if (dist1 < second_min_dist) {
        second_min_dist = dist1;
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
//-----------------------------------------------------------------------------------------------
/* reordering + warp divergence */
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter2_2(
    const uint32_t offset, const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ passed, const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, 
    //float *__restrict__ min_dists, uint32_t *__restrict__ nearests,
    float *__restrict__ second_min_dists,
    int32_t *__restrict__ mark_threads, uint16_t *__restrict__ calculate_data_point) {

  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= d_passed_number){
    return;
  }

  sample = passed[sample];
  if (calculate_data_point[sample] == 0) {
    return;
  }

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];  

  uint32_t valid_centroids = calculate_data_point[sample];
  
    
  for (int16_t i = 0; i < valid_centroids ; i++) {
    uint32_t c = mark_threads[i * d_samples_size + sample]; 

    // float dist = METRIC<M, F>::distance_t(samples, centroids + (c) * d_features_size, d_samples_size, sample + offset);
    float dist = 0;

    #pragma unroll 4
    for (uint16_t f = 0; f < d_features_size; ++f) {
      float d = samples[d_samples_size * f + sample] - centroids[c * d_features_size + f];
        dist += d * d;
    }
    float dist1 = sqrt(dist);

    if (dist1 < min_dist) {
      second_min_dist = min_dist;
      min_dist = dist1;
      nearest = c;
    } else if (dist1 < second_min_dist) {
      second_min_dist = dist1;
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

//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter_threshold(
    uint16_t *__restrict__ calculate_data_point, float *current_skip_ratio, uint32_t *partition_threshold, uint32_t *calculate_sum){
 
  //uint32_t sum = 0;
 
  // for(uint32_t i = 0 ; i < d_samples_size ; i+=20){
  //   sum += calculate_data_point[i]; // total calculation
  // }
  // /* Claculate Skip Ratio */
  //float skip_ratio1 = 1.0f - (static_cast<float>(sum) / ((d_passed_number/20) * d_clusters_size)); // correct
  // //printf("sum = %u\n", sum);
  // //printf("calculate_sum = %u\n", *calculate_sum);
  float skip_ratio1 = 1.0f - (static_cast<float>(*calculate_sum) / static_cast<float>(d_passed_number * d_clusters_size)); // correct
  // //printf("skip_ratio1 = %0.5f\n", skip_ratio1);
   *current_skip_ratio = skip_ratio1; // correct

}


//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter_partition(
    const uint32_t *__restrict__ passed,
    uint32_t *__restrict__ mark_cpu, uint32_t *__restrict__ mark_gpu, float *current_skip_ratio,
    uint32_t *partition_threshold, uint32_t *cpu_sum, uint32_t *gpu_sum, uint16_t *__restrict__ calculate_data_point,
    uint32_t *d_mark_cpu_number, uint32_t *d_mark_gpu_number, int8_t flag){

    int32_t sample = blockIdx.x * blockDim.x + threadIdx.x; 
    if (sample >= d_passed_number) {
        return;  
    }

    sample = passed[sample]; 

    if(flag == 3){
      //mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
      if(*current_skip_ratio < 1 && *current_skip_ratio >= 0.6){
        if(sample < 1200000){
            mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
          }
          else{
            mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
        }
      
      } 
      else if(*current_skip_ratio < 0.6 && *current_skip_ratio >= 0.2){
        if(sample < 1000000){
            mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
          }
          else{
            mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
        }
      
      } 

       else {
        if(sample < 800000){
            mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
          }
          else{
            mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
        }
      
      } 
    }
    else if(flag == 4){
      if(*current_skip_ratio < 1 && *current_skip_ratio >= 0.8){
        if(sample < 1200000){
            mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
          }
          else{
            mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
        }
      
      } 
      else if(*current_skip_ratio < 0.8 && *current_skip_ratio >= 0.6){
        if(sample < 700000){
            mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
          }
          else{
            mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
        }
      
      } 

       else {
        if(sample < 600000){
            mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
          }
          else{
            mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
        }
      
      } 
  
      
   }

    
    
  //---------------------------------------------------------------------------------

      
      //mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
      //mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
      //printf("partition_threshold = %u\n", *partition_threshold);
  
//---------------------------------------------------------------------------------

}
//--------------------------------------------------------------------------------------------------------------------------
template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter_partition_2(
    const uint32_t *__restrict__ passed,
    uint32_t *__restrict__ mark_cpu, uint32_t *__restrict__ mark_gpu, float *current_skip_ratio,
    uint32_t *partition_threshold, 
    uint32_t *d_mark_cpu_number, uint32_t *d_mark_gpu_number, int8_t flag){

    int32_t sample = blockIdx.x * blockDim.x + threadIdx.x; 
    if (sample >= d_passed_number) {
        return;  
    }

    sample = passed[sample]; 

  
    
      //if ( *current_skip_ratio < 1 && *current_skip_ratio >= 0.95 ){
        if(sample < 460000){
            mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
            //atomicAdd(cpu_sum, calculate_data_point[sample]);
        }else{
            mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
            //atomicAdd(gpu_sum, calculate_data_point[sample]);
        }
      //}
      
      
      

    //}
  //---------------------------------------------------------------------------------

      
      //mark_gpu[atomicAggInc(d_mark_gpu_number)] = sample;
      //mark_cpu[atomicAggInc(d_mark_cpu_number)] = sample;
      //printf("partition_threshold = %u\n", *partition_threshold);
  
//---------------------------------------------------------------------------------

}
//-------------------------------------------------------------------------------------------------------------------


template <KMCUDADistanceMetric M, typename F>
__global__ void kmeans_yy_local_filter2_3(
    const uint32_t offset, const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ mark_gpu, 
    const uint32_t *__restrict__ passed, 
    const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists, uint32_t *d_mark_gpu_number, int32_t *__restrict__ mark_threads,
    const uint16_t *__restrict__ calculate_data_point,
    uint32_t *partition_threshold

   ) {
  volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= *d_mark_gpu_number) {
  //if (sample >= d_passed_number) {
    return;
  }
  // if((sample) < *partition_threshold){
  //   return;
  // }

  sample = mark_gpu[sample];

  //sample = passed[sample];

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];


  for(uint32_t i = 0 ; i < 32 ; i++){
    int32_t mask = mark_threads[sample * 32 + i]; 

        if(mask == 0 && i < 31){
          uint32_t new_mask = 0;
          for(uint32_t j = 1 ; j < (32-i); j++){ 
              new_mask = mark_threads[sample * 32 + i + j];
              if(new_mask != 0){
                  i += j;
                  mask = new_mask;
                  break;
              }
          }
        }
        
        while(mask!= 0){
          int first_bit_pos = __ffs(mask) - 1; 
          uint32_t c = i * 32 + (first_bit_pos); 

          double dist = 0;
          #pragma unroll 4
          for (uint16_t f = 0; f < d_features_size; ++f) {
              double d = samples[d_samples_size * f + sample] - centroids[c * d_features_size + f];
              dist += d * d;
          }
          float dist_1 = sqrt(dist);

          if (dist_1 < min_dist) {
              second_min_dist = min_dist;
              min_dist = dist_1;
              nearest = c;
          } else if (dist_1 < second_min_dist) {
              second_min_dist = dist_1;
          }
          mask &= ~(1 << first_bit_pos);
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
__global__ void kmeans_yy_local_filter3(
    const uint32_t offset, const uint32_t length, const float *__restrict__ samples,
    const uint32_t *__restrict__ mark_gpu, 
    const uint32_t *__restrict__ passed, 
    const float *__restrict__ centroids,
    const uint32_t *__restrict__ groups, const float *__restrict__ drifts,
    uint32_t *__restrict__ assignments, float *__restrict__ bounds, float *__restrict__ second_min_dists, uint32_t *d_mark_gpu_number,
    uint32_t *partition_threshold
   ) {

   volatile uint32_t sample = blockIdx.x * blockDim.x + threadIdx.x;

  if (sample >= *d_mark_gpu_number) {
  //if (sample >= d_passed_number){
    return;
  }

  sample = mark_gpu[sample];
  //sample = passed[sample];
  // if((sample) < *partition_threshold){
  //   return;
  // }

  float upper_bound = bounds[sample];
  uint32_t cluster = assignments[sample];
  uint32_t nearest = cluster;
  float min_dist = upper_bound;
  float second_min_dist = second_min_dists[sample];
  uint32_t doffset = d_clusters_size * d_features_size;

  for (uint32_t c = 0; c < d_clusters_size ; c++) {
    if (c == cluster) {
        continue;
      }
      uint32_t group = groups[c];
      if (group >= d_yy_groups_size) {
        // this may happen if the centroid is insane (NaN)
        continue;
      }
      float lower_bound = bounds[static_cast<uint64_t>(length) * (1 + group) + sample];
      if (lower_bound >= upper_bound) {
        if (lower_bound < second_min_dist) {
          second_min_dist = lower_bound;
        }
        continue;
      }
      lower_bound += drifts[group] - drifts[doffset + c];
      if (second_min_dist < lower_bound) {
        continue;
      }

      float dist = 0;

      #pragma unroll 4
      for (uint16_t f = 0; f < d_features_size; ++f) {
        /* 1 */
        float d = samples[d_samples_size * f + sample] - centroids[c * d_features_size + f];

          dist += d * d;
      }
      float dist_1 = sqrt(dist);

      if (dist_1 < min_dist) {
        second_min_dist = min_dist;
        min_dist = dist_1;
        nearest = c;
      } else if (dist_1 < second_min_dist) {
        second_min_dist = dist_1;
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

// 전력 측정 시작 함수
void capture_power_metrics_in_interval_start(int iteration) {
    char start_cmd[256];
    snprintf(start_cmd, sizeof(start_cmd), "sudo tegrastats --interval 100 --logfile /home/du6293/kmcuda/src/Power_2M_256/Kubism/tegra_log_%d.txt &", iteration);
    if (system(start_cmd) == -1) {  // tegrastats 시작
        printf("Failed to start tegrastats\n");
    }
}

void capture_power_metrics_in_interval_end(int iteration) {

    if (system("sudo pkill tegrastats") == -1) {  // tegrastats 종료
        printf("Failed to stop tegrastats\n");
        return;
    }

    char log_file[256];
    snprintf(log_file, sizeof(log_file), "/home/du6293/kmcuda/src/Power_2M_256/Kubism/tegra_log_%d.txt", iteration);

    // 로그 파일에 쓰기 권한 추가
    char chmod_cmd[300];  // 버퍼 크기를 충분히 크게 설정
    int chmod_length = snprintf(chmod_cmd, sizeof(chmod_cmd), "sudo chmod 666 %s", log_file);
    
    // 타입을 일치시키기 위해 sizeof는 size_t로 캐스팅하여 비교
    if (chmod_length >= (int)sizeof(chmod_cmd)) {
        printf("Error: chmod command exceeds buffer size.\n");
        return;
    }

    if (system(chmod_cmd) == -1) {
        printf("Failed to set write permission for log file: %s\n", log_file);
        return;
    }

    // 로그 파일을 읽기 모드로 열기
    FILE *file = fopen(log_file, "r");
    if (file == NULL) {
        printf("Failed to open log file: %s\n", log_file);
        return;
    }

    // // 두 메트릭만 기록할 새로운 로그 파일을 쓰기 모드로 열기
    // char real_log_file[256];
    // snprintf(real_log_file, sizeof(real_log_file), "/home/du6293/kmcuda/src/Power_1M_256/tegra_real_log_%d.txt", iteration);

    // FILE *new_file = fopen(real_log_file, "w");
    // if (new_file == NULL) {
    //     printf("Failed to open new log file for writing: %s\n", real_log_file);
    //     fclose(file);
    //     return;
    // }

    // char line[1024];
    // while (fgets(line, sizeof(line), file) != NULL) {
    //     char *gpu_pos = strstr(line, "VDD_GPU_SOC");
    //     char *cpu_pos = strstr(line, "VDD_CPU_CV");

    //     if (gpu_pos && cpu_pos) {
    //         // 두 메트릭만 새로운 로그 파일에 기록
    //         fprintf(new_file, "%.15s %.15s\n", gpu_pos, cpu_pos);
    //     }
    // }

    // fclose(file);
    // fclose(new_file);

    // printf("Two metrics saved in: %s\n", real_log_file);
}


static int check_changed(int iter, float tolerance, uint32_t h_samples_size, const std::vector<int> &devs, int32_t verbosity) {
  //printf("check changed!\n");
  uint32_t overall_changed = 0;
  //uint32_t overall_changed = h_changed_number_cpu;
  //FOR_EACH_DEV(
    uint32_t my_changed = 0;
    CUCH(cudaMemcpyFromSymbol(&my_changed, d_changed_number, sizeof(my_changed)), kmcudaMemoryCopyError);  // device -> host
    overall_changed += my_changed;
  //);
  INFO("Iteration %d: %" PRIu32 " Reassignments\n", iter, overall_changed);
  //printf("\n");
  if (overall_changed <= tolerance * h_samples_size) {   // 0.009 * 1000000 = 9000
    return -1;
  }
  assert(overall_changed <= h_samples_size);
  uint32_t zero = 0;
  //FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbolAsync(d_changed_number, &zero, sizeof(zero)), kmcudaMemoryCopyError);  // host -> device
    cudaDeviceSynchronize(); 
  //);
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
    float tolerance, uint32_t h_yy_groups_size, uint32_t h_samples_size,
    uint32_t h_clusters_size, uint16_t h_features_size, KMCUDADistanceMetric metric,
    const std::vector<int> &devs, int fp16x2, int32_t verbosity,
    const udevptrs<float> &samples, udevptrs<float> *centroids,
    udevptrs<uint32_t> *ccounts, udevptrs<uint32_t> *assignments_prev,
    udevptrs<uint32_t> *assignments, udevptrs<uint32_t> *assignments_yy,
    udevptrs<float> *centroids_yy, //udevptrs<float> *bounds_yy,
    udevptrs<float> *drifts_yy, udevptrs<uint32_t> *passed_yy) {
  //---------------------------------------------------------------------------------
  float* bounds_yy;
  uint32_t yyb_size =  h_samples_size * (h_yy_groups_size + 1) * sizeof(float);
  //cudaMallocManaged(&bounds_yy, yyb_size);
  cudaHostAlloc((void**)&bounds_yy, yyb_size, cudaHostAllocDefault);
  
  // uint32_t *assignment_yy;
  // cudaMallocManaged(&assignment_yy, h_samples_size * sizeof(uint32_t));
  uint32_t* assignment_yy;
  cudaHostAlloc((void**)&assignment_yy, h_samples_size * sizeof(uint32_t), cudaHostAllocDefault);

  //---------------------------------------------------------------------------------

  // float* samples_yy;
  // cudaHostAlloc((void**)&samples_yy, h_samples_size * h_features_size * sizeof(float), cudaHostAllocDefault);
  // float* centroids_yyy;
  // cudaHostAlloc((void**)&centroids_yyy, h_clusters_size * h_features_size * sizeof(float), cudaHostAllocDefault);
  // uint32_t* assignments_yyy;
  // cudaHostAlloc((void**)&assignments_yyy, h_clusters_size * h_features_size * sizeof(uint32_t), cudaHostAllocDefault);
  // float* drifts_yyy;
  //  cudaHostAlloc((void**)&drifts_yyy, h_clusters_size * (h_features_size + 1) * sizeof(float), cudaHostAllocDefault);




  //---------------------------------------------------------------------------------

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
      centroids, ccounts, assignments_prev, assignments,
      &iter));
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
  RETERR(prepare_mem(h_samples_size, h_clusters_size, true, devs, verbosity, ccounts, 
  assignments, assignments_prev, &shmem_sizes));

  
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

  float     *h_samples        = (float*)malloc(h_samples_size * h_features_size * sizeof(float)); 
  //float     *h_centroids      = (float*)malloc(h_clusters_size * h_features_size * sizeof(float));

  FOR_EACH_DEVI(
    cudaMemcpy(h_samples, (samples)[devi].get(), h_samples_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_centroids, (*centroids)[devi].get(), h_clusters_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
  );

  //------------------------------------------------------iter-------------------------------------------------------------
  for (; ; iter++) {
    if (!refresh) { // refresh: decine whether initialize bounds -> do not initialize
      int status = check_changed(iter-1, tolerance, h_samples_size, devs, verbosity);
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
      );
      if (1.f - (h_passed_number + 0.f) / h_samples_size < YINYANG_REFRESH_EPSILON) {
        refresh = true;
        float cnt = 1.f - (h_passed_number + 0.f) / h_samples_size < YINYANG_REFRESH_EPSILON;
        printf("start initialize! cnt = %f\n", cnt);
       
      }
      h_passed_number = 0;
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

        cudaEvent_t start1, stop1;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        cudaEventRecord(start1);      
        flag = 3;/////////////////////////////////
        cnt1 = 0;
        cnt2 = 0;
        skip_threshold = 0;
        dim3 sigrid(upper(length, siblock.x), 1, 1);
        KERNEL_SWITCH(kmeans_yy_init, <<<sigrid, siblock, shmem_sizes[devi]>>>(
              offset, length,
              reinterpret_cast<const F*>(samples[devi].get()),
              reinterpret_cast<const F*>((*centroids)[devi].get()),
              (*assignments)[devi].get() + offset,
              (*assignments_yy)[devi].get(), 
              //(*bounds_yy)[devi].get()
              bounds_yy));
        cudaDeviceSynchronize();
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start1, stop1);
        printf("(1) Initialization kernel execution time: %f ms\n", milliseconds1);
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

      cudaEvent_t start2, stop2;
      cudaEventCreate(&start2);
      cudaEventCreate(&stop2);
      cudaEventRecord(start2);

      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_adjust, <<<cgrid, cblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()),
          (*assignments_prev)[devi].get(), (*assignments)[devi].get() + offset,
          reinterpret_cast<F*>((*centroids)[devi].get()), (*ccounts)[devi].get()));
      cudaDeviceSynchronize();
      cudaEventRecord(stop2);
      cudaEventSynchronize(stop2);

      float milliseconds2 = 0;
      cudaEventElapsedTime(&milliseconds2, start2, stop2);
      printf("\n");
      printf("[Current Iteration = %d]\n", iter);
      //printf("(2) Centroid Adjust kernel execution time: %f ms\n", milliseconds2);

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


      cudaEvent_t start3, stop3;
      cudaEventCreate(&start3);
      cudaEventCreate(&stop3);
      cudaEventRecord(start3);  

      dim3 cgrid(upper(length, cblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_calc_drifts, <<<cgrid, cblock>>>(
          offset, length, reinterpret_cast<const F*>((*centroids)[devi].get()),
          reinterpret_cast<F*>((*drifts_yy)[devi].get())));
      cudaDeviceSynchronize();
      cudaEventRecord(stop3);
      cudaEventSynchronize(stop3);

      float milliseconds3 = 0;
      cudaEventElapsedTime(&milliseconds3, start3, stop3);
     // printf("(3) Drift Calculation kernel execution time: %f ms\n", milliseconds3);

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
      

      cudaEvent_t start4, stop4;
      cudaEventCreate(&start4);
      cudaEventCreate(&stop4);

      cudaEventRecord(start4);
      dim3 ggrid(upper(length, gblock.x), 1, 1);
      kmeans_yy_find_group_max_drifts<<<ggrid, gblock, shmem_sizes[devi]>>>(
          offset, length, (*assignments_yy)[devi].get(),
          (*drifts_yy)[devi].get());
      cudaDeviceSynchronize();
      cudaEventRecord(stop4);
      cudaEventSynchronize(stop4);
      float milliseconds4 = 0;
      cudaEventElapsedTime(&milliseconds4, start4, stop4);
      //printf("(4) Group Max Drifts kernel execution time: %f ms\n", milliseconds4);
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
      );
//--------------------------------------------------------------------------------yinyang 5 ~ 7-------------------------------------------------------------------------------
   
    FOR_EACH_DEVI(
      //uint32_t h_distance_number = 0; // initialization
      uint32_t offset, length;
      std::tie(offset, length) = plans[devi];
      if (length == 0) {
        continue;
      }
      
      cudaEvent_t start5, stop5;
      cudaEventCreate(&start5);
      cudaEventCreate(&stop5);
      cudaEventRecord(start5);
      dim3 sggrid(upper(length, sgblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_global_filter, <<<sggrid, sgblock>>>(
          offset, length,
          reinterpret_cast<const F*>(samples[devi].get()),
          reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), 
          (*drifts_yy)[devi].get(),
          //drifts_yyy,
          (*assignments)[devi].get() + offset, 
          (*assignments_prev)[devi].get() + offset,
          bounds_yy,
          //(*bounds_yy)[devi].get(),
          (*passed_yy)[devi].get()));
      cudaDeviceSynchronize();
      cudaEventRecord(stop5);
      cudaEventSynchronize(stop5);

      float milliseconds5 = 0;
      cudaEventElapsedTime(&milliseconds5, start5, stop5);
     // printf("(5) Global Filter kernel execution time: %f ms\n", milliseconds5);

      cudaEventDestroy(start5);
      cudaEventDestroy(stop5);

      uint32_t global_to_local;
      CUCH(cudaMemcpyFromSymbol(&global_to_local, d_passed_number, sizeof(uint32_t)), kmcudaMemoryCopyError);
      printf("(5) Iteration %d Global Filter to Local Filter: %u\n", iter, global_to_local);     
      //printf("h_samples_size = %u h_clusters_size = %u, h_features_size = %u\n", h_samples_size, h_clusters_size, h_features_size); 

//--------------------------------------------------------------------------------------------------------------------------------------------------
      // std::cout << "Start capturing power metrics..." << std::endl;
      // capture_power_metrics_in_interval_start(iter);

//-------------------------------------------------------------------------------------------------------------------------------------------------
      int32_t *mark_threads_yy;
      cudaHostAlloc(&mark_threads_yy, ( h_samples_size) * (h_clusters_size) * sizeof(int32_t), cudaHostAllocMapped);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
      }
      uint16_t *calculate_data_point_yy;
      cudaHostAlloc(&calculate_data_point_yy, ( h_samples_size) * sizeof(uint16_t), cudaHostAllocMapped);

      float *second_min_dists_yy;
      cudaHostAlloc(&second_min_dists_yy, static_cast<size_t>(h_samples_size) * sizeof(float), cudaHostAllocMapped); 

      memset(mark_threads_yy, 0, (h_samples_size) * static_cast<int32_t>(h_clusters_size) * sizeof(int32_t));
      memset(calculate_data_point_yy, 0, (h_samples_size) * sizeof(uint16_t));
      //cudaMemset(d_calculate_centroid, 0, static_cast<uint32_t>(h_clusters_size) * sizeof(uint32_t));
      memset(second_min_dists_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(float));

      uint32_t *calculate_sum;
      cudaHostAlloc((void**)&calculate_sum, sizeof(uint32_t), cudaHostAllocMapped);
      *calculate_sum = 0;

      float *current_skip_ratio;
      cudaHostAlloc((void**)&current_skip_ratio, sizeof(float), cudaHostAllocMapped);
      *current_skip_ratio = 0.0f;

      uint32_t *partition_threshold;
      cudaHostAlloc((void**)&partition_threshold, sizeof(uint32_t), cudaHostAllocMapped);
      *partition_threshold = 0;

      //cudaDeviceSynchronize();
//------------------------------------------------------------------------------------------------------------------------------------- 
    if(flag == 0){
        cudaEvent_t start11, stop11;
        cudaEventCreate(&start11);
        cudaEventCreate(&stop11);
        cudaEventRecord(start11);

        dim3 slgrid(upper(length, slblock.x), 1, 1);
        KERNEL_SWITCH(kmeans_yy_local_filter2_0_0, <<<slgrid, slblock, shmem_sizes[devi]>>>(
            offset, length, reinterpret_cast<const F*>(samples[devi].get()), (*passed_yy)[devi].get(),
            reinterpret_cast<const F*>((*centroids)[devi].get()),
            (*assignments_yy)[devi].get(), 
            (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
            //drifts_yyy,
            (*assignments)[devi].get() + offset, 
            bounds_yy,
            //(*bounds_yy)[devi].get(), 
            second_min_dists_yy,
            mark_threads_yy, calculate_data_point_yy
            ));
        cudaDeviceSynchronize();

        cudaEventRecord(stop11);
        cudaEventSynchronize(stop11);
        float milliseconds11 = 0;
        cudaEventElapsedTime(&milliseconds11, start11, stop11);
        printf("(6) Local Filter Baseline kernel execution time: %f ms\n", milliseconds11);
    
        cudaEventDestroy(start11);
        cudaEventDestroy(stop11); 
   }
    //--------------------------------------------------------------------------------------------------------------------------------------
    else{
      cudaEvent_t start6, stop6;
      cudaEventCreate(&start6);
      cudaEventCreate(&stop6);
      cudaEventRecord(start6);

      dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter1_1, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()), (*passed_yy)[devi].get(),
          reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), 
          (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
          //drifts_yyy,
          (*assignments)[devi].get() + offset, 
          bounds_yy,
          //(*bounds_yy)[devi].get(), 
          second_min_dists_yy,
          mark_threads_yy, calculate_data_point_yy, calculate_sum));
      cudaDeviceSynchronize();

      cudaEventRecord(stop6);
      cudaEventSynchronize(stop6);
      float milliseconds6 = 0;
      cudaEventElapsedTime(&milliseconds6, start6, stop6);
      printf("(6) Early Skip Ratio Marking kernel execution time: %f ms\n", milliseconds6);

      cudaEventDestroy(start6);
      cudaEventDestroy(stop6);
//-----------------------------------------------------------------------------------------------------------     
      cudaEvent_t start7, stop7;
      cudaEventCreate(&start7);
      cudaEventCreate(&stop7);
      cudaEventRecord(start7);

      //int threads_per_block = 512;  // 한 블록당 최대 1024개의 스레드
      //int blocks = (h_samples_size + threads_per_block - 1) / threads_per_block;  // 필요한 블록 수 계산
      KERNEL_SWITCH(kmeans_yy_local_filter_threshold, <<<1,h_clusters_size>>>(
        calculate_data_point_yy,
        current_skip_ratio, partition_threshold, calculate_sum));
      cudaDeviceSynchronize();

      cudaEventRecord(stop7);
      cudaEventSynchronize(stop7);
      float milliseconds7 = 0;
      cudaEventElapsedTime(&milliseconds7, start7, stop7);
      printf("(7) Calculation Skip Ratio execution time: %f ms\n", milliseconds7);
      cudaEventDestroy(start7);
      cudaEventDestroy(stop7);
      printf("Current iteration Skip Ratio = %0.5f\n", *current_skip_ratio);
 }
      

      if(cnt1 == 1 ){
        if(skip_threshold < *current_skip_ratio ) flag = 3;
        else flag = 4;
      }
      printf("flag = %u skip_threshold = %0.3f\n", flag, skip_threshold);

    if(flag == 3) {   

//---------------------------------------------------------------------------------------------------------------
      float predict_execution_time = prev_execution_time * ((static_cast<float>(1)-prev_skip_ratio)) / ((static_cast<float>(1) - *current_skip_ratio));
//----------------------------------------------------------------------------------------------------------------

      cudaEvent_t start6, stop6;
      cudaEventCreate(&start6);
      cudaEventCreate(&stop6);
      cudaEventRecord(start6);

      dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter1, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length, reinterpret_cast<const F*>(samples[devi].get()), (*passed_yy)[devi].get(),
          reinterpret_cast<const F*>((*centroids)[devi].get()),
          (*assignments_yy)[devi].get(), 
          (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
          //drifts_yyy,
          (*assignments)[devi].get() + offset, 
          bounds_yy,
          //(*bounds_yy)[devi].get(), 
          second_min_dists_yy,
          mark_threads_yy, calculate_data_point_yy));
      cudaDeviceSynchronize();

      cudaEventRecord(stop6);
      cudaEventSynchronize(stop6);
      float milliseconds6 = 0;
      cudaEventElapsedTime(&milliseconds6, start6, stop6);
      printf("(6) Local Filter Marking kernel execution time: %f ms\n", milliseconds6);
  
      cudaEventDestroy(start6);
      cudaEventDestroy(stop6); 
//----------------------------------------------------------------------------------------------------------------------
      uint32_t *cpu_sum;
      cudaHostAlloc((void**)&cpu_sum, sizeof(uint32_t), cudaHostAllocMapped);
      *cpu_sum = 0;

      uint32_t *gpu_sum;
      cudaHostAlloc((void**)&gpu_sum, sizeof(uint32_t), cudaHostAllocMapped);
      *gpu_sum = 0;

//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      cudaEvent_t start8, stop8;
      cudaEventCreate(&start8);
      cudaEventCreate(&stop8);
      cudaEventRecord(start8);

      uint32_t *mark_cpu_yy;
      cudaHostAlloc((void**)&mark_cpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);

      uint32_t *mark_gpu_yy;
      cudaHostAlloc((void**)&mark_gpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);
 
      memset(mark_cpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));
      memset(mark_gpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));

      uint32_t *mark_cpu_number;
      cudaHostAlloc(&mark_cpu_number, sizeof(uint32_t), cudaHostAllocMapped);
      
      uint32_t *mark_gpu_number;
      cudaHostAlloc(&mark_gpu_number, sizeof(uint32_t), cudaHostAllocMapped);

      memset(mark_cpu_number, 0, sizeof(uint32_t)); // Set to 0
      memset(mark_gpu_number, 0, sizeof(uint32_t)); // Set to 0

      //cudaDeviceSynchronize();
      //dim3 slgrid(upper(length, slblock.x), 1, 1);
      
      KERNEL_SWITCH(kmeans_yy_local_filter_partition, <<<slgrid, slblock>>>((*passed_yy)[devi].get(), 
                                                                            mark_cpu_yy, mark_gpu_yy, current_skip_ratio,
                                                                            partition_threshold, cpu_sum, gpu_sum, calculate_data_point_yy,
                                                                            mark_cpu_number, mark_gpu_number, flag
                                                                            ));
      cudaDeviceSynchronize();

      cudaEventRecord(stop8);
      cudaEventSynchronize(stop8);
      float milliseconds8 = 0;
      cudaEventElapsedTime(&milliseconds8, start8, stop8);
      
      printf("(8) CPU-GPU Data Point Partition kernel execution time: %f ms\n", milliseconds8);
      //printf("partition_threshold = %u\n", *partition_threshold);
      cudaEventDestroy(start8);
      cudaEventDestroy(stop8); 


      
      //printf("(9-3-1) Partition threshold = %u\n", *partition_threshold);
      printf("(9-3-2) CPU Threads = %u GPU Threads = %u Thread ratio = %.2f\n", *mark_cpu_number, *mark_gpu_number, (static_cast<float>(*mark_gpu_number)/static_cast<float>(*mark_cpu_number)));
      printf("(9-3-2) CPU calculation = %u  GPU calculation = %u \n", *cpu_sum, *gpu_sum);


//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      
      cudaEvent_t start_d2h, stop_d2h;
      cudaEventCreate(&start_d2h);
      cudaEventCreate(&stop_d2h);
      cudaEventRecord(start_d2h);
      
      //float     *h_samples        = (float*)malloc(h_samples_size * h_features_size * sizeof(float)); 
      float     *h_centroids      = (float*)malloc(h_clusters_size * h_features_size * sizeof(float));
      uint32_t  *h_assignments_yy = (uint32_t*)malloc(h_clusters_size * sizeof(uint32_t));
      float     *h_drifts_yy      = (float*)malloc((h_clusters_size) * (h_features_size + 1) * sizeof(float));
      uint32_t  *h_passed_yy        = (uint32_t*)malloc(h_samples_size * sizeof(uint32_t));
      

      cudaMemcpy(assignment_yy, (*assignments)[devi].get(), h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
      //cudaMemcpy(h_samples, (samples)[devi].get(), h_samples_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_centroids, (*centroids)[devi].get(), h_clusters_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_assignments_yy, (*assignments_yy)[devi].get(), h_clusters_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_drifts_yy, (*drifts_yy)[devi].get(), (h_clusters_size) * (h_features_size + 1) * sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(h_passed_yy, (*passed_yy)[devi].get(), (h_samples_size) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      cudaEventRecord(stop_d2h);
      cudaEventSynchronize(stop_d2h);
      float milliseconds_d2h = 0;
      cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h);
      printf("(9-3-3) Device to Host Data Transfer Time: %f ms\n", milliseconds_d2h);
      cudaEventDestroy(start_d2h);
      cudaEventDestroy(stop_d2h);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------
      cudaEvent_t kernel_start;
      cudaEvent_t kernel_stop;
      cudaEvent_t  sync_start;
      //cudaEvent_t sync_stop;
      cudaEventCreate(&kernel_start);
      cudaEventCreate(&kernel_stop);
      cudaEventCreate(&sync_start);
      // cudaEventCreate(&sync_stop);

      // GPU 커널 시작 시간을 기록
      cudaEventRecord(kernel_start);
      // GPU 커널 호출
      //dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter2_3, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length,  
          reinterpret_cast<const float*>(samples[devi].get()), 
          //samples_yy,
          mark_gpu_yy, (*passed_yy)[devi].get(),
          reinterpret_cast<const float*>((*centroids)[devi].get()),
          //centroids_yyy,
          (*assignments_yy)[devi].get(), 
          //assignments_yyy,
          (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
          //drifts_yyy,
          assignment_yy,
          //(*assignments)[devi].get() + offset,
          bounds_yy,
          //(*bounds_yy)[devi].get(),
          second_min_dists_yy, mark_gpu_number, 
          mark_threads_yy, calculate_data_point_yy,
          partition_threshold
          //d_samples_row_major, d_centroids_col_major
          ));
      //cudaDeviceSynchronize();
      cudaEventRecord(kernel_stop);
      //cudaEventSynchronize(kernel_stop);
     
      uint32_t h_changed_number_cpu = 0;
      local_filter_cpu1<float>(h_samples_size, h_clusters_size, h_features_size, h_yy_groups_size,
                              h_samples, mark_cpu_yy, h_centroids, h_assignments_yy, h_drifts_yy, 
                              assignment_yy, bounds_yy, second_min_dists_yy, 
                              mark_cpu_number, h_changed_number_cpu,
                              mark_threads_yy, calculate_data_point_yy, partition_threshold);
      cudaEventRecord(sync_start);
      cudaDeviceSynchronize();///////////////////////////////////////////////////////////////////////////////////////////////////
      //cudaEventRecord(sync_stop);
      //cudaEventSynchronize(sync_stop);////////////////////////////////////

      float kernel_time = 0;
      cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
      printf("(9-3-4) RWD kernel execution time: %f ms\n", kernel_time);

      float sync_time1 = 0;
      cudaEventElapsedTime(&sync_time1, kernel_start, sync_start);
      printf("(9-3-4) kernel start ~ sync start: %f ms\n", sync_time1);


      // CUDA 이벤트 제거
      cudaEventDestroy(kernel_start);
      cudaEventDestroy(kernel_stop);
      cudaEventDestroy(sync_start);
      //cudaEventDestroy(sync_stop);
//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
/* H2D */     
      cudaEvent_t start_h2d, stop_h2d;
      cudaEventCreate(&start_h2d);
      cudaEventCreate(&stop_h2d);
      cudaEventRecord(start_h2d);
  

      cudaMemcpy((*assignments)[devi].get(), assignment_yy, h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

      cudaDeviceSynchronize();
      cudaEventRecord(stop_h2d);
      cudaEventSynchronize(stop_h2d);
      float milliseconds_h2d = 0;
      cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);
      printf("(9-3-5) Host To Device Data Transfer Time: %f ms\n", milliseconds_h2d);
      //float total_time = milliseconds6 + milliseconds7 + milliseconds8 + milliseconds_d2h + sync_time1 + milliseconds_h2d;
      //printf("(9-3-6) Total Time: %f ms\n", total_time);
      cudaEventDestroy(start_h2d);
      cudaEventDestroy(stop_h2d);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      /* Retrieve d_changed_number from GPU */
      uint32_t h_changed_number_gpu;
      CUCH(cudaMemcpyFromSymbol(&h_changed_number_gpu, d_changed_number, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost), kmcudaMemoryCopyError); // device -> host
      cudaDeviceSynchronize();
      /* Combine CPU and GPU changed numbers */
      uint32_t total_changed_number = h_changed_number_cpu + h_changed_number_gpu;
      printf("(9-3-6) Iteration %d Total Reassignments (CPU + GPU): %u\n", iter, total_changed_number );
      CUCH(cudaMemcpyToSymbol(d_changed_number, &total_changed_number, sizeof(uint32_t), 0, cudaMemcpyHostToDevice), kmcudaMemoryCopyError); // reflect next iteration
//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      /* Current data to Prev data */
      prev_execution_time = sync_time1;
      //printf("3)prev_execution_time = %0.4f\n", prev_execution_time);
      prev_skip_ratio = *current_skip_ratio;
     // prev_global_to_local = global_to_local;
      printf("prev_skip_ratio = %0.3f, current_skip_ratio = %0.3f \n", prev_skip_ratio, *current_skip_ratio);
      

      // if(cnt1 == 0) { // iter 4, 20 
      //   skip_threshold = 0;
      //   cnt1 = 1;
      // }else{
      //   if(cnt2 == 0){  // iter 5, 21
      //     if(predict_execution_time < sync_time1){ 
      //       //flag = 4;
      //       skip_threshold = *current_skip_ratio;
      //     }          
      //     cnt2 = 1;
      //   }

      // }
//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      cudaFreeHost(cpu_sum);
      cudaFreeHost(gpu_sum);

      cudaFreeHost(mark_cpu_yy);
      cudaFreeHost(mark_gpu_yy);
      cudaFreeHost(mark_cpu_number);
      cudaFreeHost(mark_gpu_number);

      // free(h_samples);
      free(h_centroids);
      free(h_assignments_yy);
      free(h_drifts_yy);
      free(h_passed_yy);


      if (cnt1 == 0){ // kmeans_yy_init 4, 20
        skip_threshold = 0;
        cnt1 = 1;
      }else{  // Not kmeans_yy_init
        if(cnt2 == 0){    // predict_exec_time
          if(predict_execution_time < sync_time1) {
            //flag = 4;
            skip_threshold = *current_skip_ratio;
          }
          //else flag = 3;
          cnt2 = 1;
        }
      }
      
    }  
    else if (flag == 4 && h_features_size > 64){


        uint32_t *cpu_sum;
        cudaHostAlloc((void**)&cpu_sum, sizeof(uint32_t), cudaHostAllocMapped);
        *cpu_sum = 0;

        uint32_t *gpu_sum;
        cudaHostAlloc((void**)&gpu_sum, sizeof(uint32_t), cudaHostAllocMapped);
        *gpu_sum = 0;

  //---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        cudaEvent_t start8, stop8;
        cudaEventCreate(&start8);
        cudaEventCreate(&stop8);
        cudaEventRecord(start8);

        uint32_t *mark_cpu_yy;
        cudaHostAlloc((void**)&mark_cpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);

        uint32_t *mark_gpu_yy;
        cudaHostAlloc((void**)&mark_gpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);
  
        memset(mark_cpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));
        memset(mark_gpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));

        uint32_t *mark_cpu_number;
        cudaHostAlloc(&mark_cpu_number, sizeof(uint32_t), cudaHostAllocMapped);
        
        uint32_t *mark_gpu_number;
        cudaHostAlloc(&mark_gpu_number, sizeof(uint32_t), cudaHostAllocMapped);

        memset(mark_cpu_number, 0, sizeof(uint32_t)); // Set to 0
        memset(mark_gpu_number, 0, sizeof(uint32_t)); // Set to 0   
//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    

        //cudaDeviceSynchronize();
        dim3 slgrid(upper(length, slblock.x), 1, 1);
        KERNEL_SWITCH(kmeans_yy_local_filter_partition, <<<slgrid, slblock>>>((*passed_yy)[devi].get(), 
                                                                              mark_cpu_yy, mark_gpu_yy, current_skip_ratio,
                                                                              partition_threshold, cpu_sum, gpu_sum, calculate_data_point_yy,
                                                                              mark_cpu_number, mark_gpu_number, flag));
        cudaDeviceSynchronize();

        cudaEventRecord(stop8);
        cudaEventSynchronize(stop8);
        float milliseconds8 = 0;
        cudaEventElapsedTime(&milliseconds8, start8, stop8);
        printf("(8) CPU-GPU Data Point Partition kernel execution time: %f ms\n", milliseconds8);
        //printf("partition_threshold = %u\n", *partition_threshold);
        cudaEventDestroy(start8);
        cudaEventDestroy(stop8); 


        printf("(9-4-1) Skip Ratio = %.5f\n", *current_skip_ratio);
        //printf("(9-3-1) Partition threshold = %u\n", *partition_threshold);
        printf("(9-4-2) CPU Threads = %u GPU Threads = %u Thread ratio = %.2f\n", *mark_cpu_number, *mark_gpu_number, (static_cast<float>(*mark_gpu_number)/static_cast<float>(*mark_cpu_number)));
        printf("(9-4-2) CPU calculation = %u  GPU calculation = %u \n", *cpu_sum, *gpu_sum);


//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      cudaEvent_t start_d2h, stop_d2h;
      cudaEventCreate(&start_d2h);
      cudaEventCreate(&stop_d2h);
      cudaEventRecord(start_d2h);
      
      //float     *h_samples        = (float*)malloc(h_samples_size * h_features_size * sizeof(float));
      float     *h_centroids      = (float*)malloc(h_clusters_size * h_features_size * sizeof(float));
      uint32_t  *h_assignments_yy = (uint32_t*)malloc(h_clusters_size * sizeof(uint32_t));
      float     *h_drifts_yy      = (float*)malloc((h_clusters_size) * (h_features_size + 1) * sizeof(float));
      //uint32_t  *h_passed_yy        = (uint32_t*)malloc(h_samples_size * sizeof(uint32_t));
      
      cudaMemcpy(assignment_yy, (*assignments)[devi].get(), h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

     // cudaMemcpy(h_samples, (samples)[devi].get(), h_samples_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
   
      cudaMemcpy(h_centroids, (*centroids)[devi].get(), h_clusters_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(h_assignments_yy, (*assignments_yy)[devi].get(), h_clusters_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

      cudaMemcpy(h_drifts_yy, (*drifts_yy)[devi].get(), (h_clusters_size) * (h_features_size + 1) * sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(h_passed_yy, (*passed_yy)[devi].get(), (h_samples_size) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

      //cudaDeviceSynchronize();

      cudaEventRecord(stop_d2h);
      cudaEventSynchronize(stop_d2h);
      float milliseconds_d2h = 0;
      cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h);
      printf("(9-4-3) Device to Host Data Transfer Time: %f ms\n", milliseconds_d2h);
      cudaEventDestroy(start_d2h);
      cudaEventDestroy(stop_d2h);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      //RETERR(cuda_transpose(h_clusters_size, h_features_size, true, devs, verbosity, centroids));
       cudaEvent_t kernel_start;
       cudaEvent_t kernel_stop;
       cudaEvent_t  sync_start;
      // cudaEvent_t sync_stop;
       cudaEventCreate(&kernel_start);
       cudaEventCreate(&kernel_stop);
       cudaEventCreate(&sync_start);
      // cudaEventCreate(&sync_stop);

      // // GPU 커널 시작 시간을 기록
       cudaEventRecord(kernel_start);
      // GPU 커널 호출
      //dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter3, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length,  
          reinterpret_cast<const float*>(samples[devi].get()), 
          //samples_yy,
          mark_gpu_yy, (*passed_yy)[devi].get(),
          reinterpret_cast<const float*>((*centroids)[devi].get()),
          //centroids_yyy,
          (*assignments_yy)[devi].get(), 
          //assignments_yyy,
          (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
          //drifts_yyy,
          assignment_yy,
          //(*assignments)[devi].get() + offset,
          bounds_yy,
          //(*bounds_yy)[devi].get(),
          second_min_dists_yy, mark_gpu_number,
          partition_threshold

          //calculate_data_point_yy,
          //d_samples_row_major, d_centroids_col_major
          ));
      //cudaDeviceSynchronize();
      cudaEventRecord(kernel_stop);
      //cudaEventSynchronize(kernel_stop);
     
      uint32_t h_changed_number_cpu = 0;
      local_filter_cpu2<float>(h_samples_size, h_clusters_size, h_features_size, h_yy_groups_size,
                              h_samples, mark_cpu_yy, h_centroids, h_assignments_yy, h_drifts_yy, 
                              assignment_yy, bounds_yy, second_min_dists_yy, 
                              mark_cpu_number, h_changed_number_cpu);
       cudaEventRecord(sync_start);
       cudaDeviceSynchronize();///////////////////////////////////////////////////////////////////////////////////////////////////
       //cudaEventRecord(sync_stop);
       //cudaEventSynchronize(sync_stop);////////////////////////////////////

      float kernel_time = 0;
      cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
      printf("(9-4-4) Only Heterogeneous kernel execution time: %f ms\n", kernel_time);

      float sync_time1 = 0;
      cudaEventElapsedTime(&sync_time1, kernel_start, sync_start);
      printf("(9-4-4) Only Heterogeneous kernel start ~ sync start: %f ms\n", sync_time1);


      //float kernel_sync_time1 = 0 ; 
      //cudaEventElapsedTime(&kernel_sync_time1, kernel_start, sync_stop);
      //printf("(9-2) kernel start ~ sync stop: %f ms\n", kernel_sync_time1);



      // CUDA 이벤트 제거
      cudaEventDestroy(kernel_start);
      cudaEventDestroy(kernel_stop);
      cudaEventDestroy(sync_start);
      //cudaEventDestroy(sync_stop);
 
      
//----------------------------------------------------------------------------------------------------------------------------      
/* H2D */     
      cudaEvent_t start_h2d, stop_h2d;
      cudaEventCreate(&start_h2d);
      cudaEventCreate(&stop_h2d);
      cudaEventRecord(start_h2d);

      cudaMemcpy((*assignments)[devi].get(), assignment_yy, h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

      cudaDeviceSynchronize();
      cudaEventRecord(stop_h2d);
      cudaEventSynchronize(stop_h2d);
      float milliseconds_h2d = 0;
      cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);
      printf("(9-4-5) Host To Device Data Transfer Time: %f ms\n", milliseconds_h2d);
      //float total_time = milliseconds6 + milliseconds7 + milliseconds8 + milliseconds_d2h + sync_time1 + milliseconds_h2d;
      //printf("(9-4-6) Total Time: %f ms\n", total_time);
      cudaEventDestroy(start_h2d);
      cudaEventDestroy(stop_h2d);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      /* Retrieve d_changed_number from GPU */
      uint32_t h_changed_number_gpu;
      CUCH(cudaMemcpyFromSymbol(&h_changed_number_gpu, d_changed_number, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost), kmcudaMemoryCopyError); // device -> host
      //printf("(9-2-4) (GPU) Changed Number: %u\n", h_changed_number_gpu);
      //cudaDeviceSynchronize();
      /* Combine CPU and GPU changed numbers */
      uint32_t total_changed_number = h_changed_number_cpu + h_changed_number_gpu;
      printf("(9-4-6) Iteration %d Total Reassignments (CPU + GPU): %u\n", iter, total_changed_number );
      CUCH(cudaMemcpyToSymbol(d_changed_number, &total_changed_number, sizeof(uint32_t), 0, cudaMemcpyHostToDevice), kmcudaMemoryCopyError); // reflect next iteration
//-------------------------------------------------------------------------------------------------------------------------
      prev_execution_time = sync_time1;
      prev_skip_ratio = *current_skip_ratio;

      cudaFreeHost(cpu_sum);
      cudaFreeHost(gpu_sum);

      cudaFreeHost(mark_cpu_yy);
      cudaFreeHost(mark_gpu_yy);
      cudaFreeHost(mark_cpu_number);
      cudaFreeHost(mark_gpu_number);

     // free(h_samples);
      free(h_centroids);
      free(h_assignments_yy);
      free(h_drifts_yy);
      //free(h_passed_yy);


    

    }

//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
  else if (flag == 4 && h_features_size <= 64 ){


        uint32_t *cpu_sum;
        cudaHostAlloc((void**)&cpu_sum, sizeof(uint32_t), cudaHostAllocMapped);
        *cpu_sum = 0;

        uint32_t *gpu_sum;
        cudaHostAlloc((void**)&gpu_sum, sizeof(uint32_t), cudaHostAllocMapped);
        *gpu_sum = 0;

  //---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        cudaEvent_t start8, stop8;
        cudaEventCreate(&start8);
        cudaEventCreate(&stop8);
        cudaEventRecord(start8);

        uint32_t *mark_cpu_yy;
        cudaHostAlloc((void**)&mark_cpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);

        uint32_t *mark_gpu_yy;
        cudaHostAlloc((void**)&mark_gpu_yy, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t), cudaHostAllocMapped);
  
        memset(mark_cpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));
        memset(mark_gpu_yy, 0, static_cast<uint32_t>(h_samples_size) * sizeof(uint32_t));

        uint32_t *mark_cpu_number;
        cudaHostAlloc(&mark_cpu_number, sizeof(uint32_t), cudaHostAllocMapped);
        
        uint32_t *mark_gpu_number;
        cudaHostAlloc(&mark_gpu_number, sizeof(uint32_t), cudaHostAllocMapped);

        memset(mark_cpu_number, 0, sizeof(uint32_t)); // Set to 0
        memset(mark_gpu_number, 0, sizeof(uint32_t)); // Set to 0

        //cudaDeviceSynchronize();
        dim3 slgrid(upper(length, slblock.x), 1, 1);
        KERNEL_SWITCH(kmeans_yy_local_filter_partition_2, <<<slgrid, slblock>>>((*passed_yy)[devi].get(), 
                                                                              mark_cpu_yy, mark_gpu_yy, current_skip_ratio,
                                                                              partition_threshold,
                                                                              mark_cpu_number, mark_gpu_number, flag));
        cudaDeviceSynchronize();

        cudaEventRecord(stop8);
        cudaEventSynchronize(stop8);
        float milliseconds8 = 0;
        cudaEventElapsedTime(&milliseconds8, start8, stop8);
        printf("(8) CPU-GPU Data Point Partition kernel execution time: %f ms\n", milliseconds8);
        //printf("partition_threshold = %u\n", *partition_threshold);
        cudaEventDestroy(start8);
        cudaEventDestroy(stop8); 


        printf("(just hetero) Skip Ratio = %.5f\n", *current_skip_ratio);
        //printf("(9-3-1) Partition threshold = %u\n", *partition_threshold);
        printf("(just hetero) CPU Threads = %u GPU Threads = %u Thread ratio = %.2f\n", *mark_cpu_number, *mark_gpu_number, (static_cast<float>(*mark_gpu_number)/static_cast<float>(*mark_cpu_number)));
        printf("(just hetero) CPU calculation = %u  GPU calculation = %u \n", *cpu_sum, *gpu_sum);


//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    

//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      cudaEvent_t start_d2h, stop_d2h;
      cudaEventCreate(&start_d2h);
      cudaEventCreate(&stop_d2h);
      cudaEventRecord(start_d2h);
      
      //float     *h_samples        = (float*)malloc(h_samples_size * h_features_size * sizeof(float));
      float     *h_centroids      = (float*)malloc(h_clusters_size * h_features_size * sizeof(float));
      uint32_t  *h_assignments_yy = (uint32_t*)malloc(h_clusters_size * sizeof(uint32_t));
      float     *h_drifts_yy      = (float*)malloc((h_clusters_size) * (h_features_size + 1) * sizeof(float));
      //uint32_t  *h_passed_yy        = (uint32_t*)malloc(h_samples_size * sizeof(uint32_t));
      
      cudaMemcpy(assignment_yy, (*assignments)[devi].get(), h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

     // cudaMemcpy(h_samples, (samples)[devi].get(), h_samples_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);
   
      cudaMemcpy(h_centroids, (*centroids)[devi].get(), h_clusters_size * h_features_size * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(h_assignments_yy, (*assignments_yy)[devi].get(), h_clusters_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

      cudaMemcpy(h_drifts_yy, (*drifts_yy)[devi].get(), (h_clusters_size) * (h_features_size + 1) * sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(h_passed_yy, (*passed_yy)[devi].get(), (h_samples_size) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

      //cudaDeviceSynchronize();

      cudaEventRecord(stop_d2h);
      cudaEventSynchronize(stop_d2h);
      float milliseconds_d2h = 0;
      cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h);
      printf("(9-4-3) Device to Host Data Transfer Time: %f ms\n", milliseconds_d2h);
      cudaEventDestroy(start_d2h);
      cudaEventDestroy(stop_d2h);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      //RETERR(cuda_transpose(h_clusters_size, h_features_size, true, devs, verbosity, centroids));
       cudaEvent_t kernel_start;
       cudaEvent_t kernel_stop;
       cudaEvent_t  sync_start;
      // cudaEvent_t sync_stop;
       cudaEventCreate(&kernel_start);
       cudaEventCreate(&kernel_stop);
       cudaEventCreate(&sync_start);
      // cudaEventCreate(&sync_stop);

      // // GPU 커널 시작 시간을 기록
       cudaEventRecord(kernel_start);
      // GPU 커널 호출
      //dim3 slgrid(upper(length, slblock.x), 1, 1);
      KERNEL_SWITCH(kmeans_yy_local_filter3, <<<slgrid, slblock, shmem_sizes[devi]>>>(
          offset, length,  
          reinterpret_cast<const float*>(samples[devi].get()), 
          //samples_yy,
          mark_gpu_yy, (*passed_yy)[devi].get(),
          reinterpret_cast<const float*>((*centroids)[devi].get()),
          //centroids_yyy,
          (*assignments_yy)[devi].get(), 
          //assignments_yyy,
          (*drifts_yy)[devi].get(),    // assignments_yy equal to groups
          //drifts_yyy,
          assignment_yy,
          //(*assignments)[devi].get() + offset,
          bounds_yy,
          //(*bounds_yy)[devi].get(),
          second_min_dists_yy, mark_gpu_number,
          partition_threshold

          //calculate_data_point_yy,
          //d_samples_row_major, d_centroids_col_major
          ));
      //cudaDeviceSynchronize();
      cudaEventRecord(kernel_stop);
      //cudaEventSynchronize(kernel_stop);
     
      uint32_t h_changed_number_cpu = 0;
      local_filter_cpu2<float>(h_samples_size, h_clusters_size, h_features_size, h_yy_groups_size,
                              h_samples, mark_cpu_yy, h_centroids, h_assignments_yy, h_drifts_yy, 
                              assignment_yy, bounds_yy, second_min_dists_yy, 
                              mark_cpu_number, h_changed_number_cpu);
       cudaEventRecord(sync_start);
       cudaDeviceSynchronize();///////////////////////////////////////////////////////////////////////////////////////////////////
       //cudaEventRecord(sync_stop);
       //cudaEventSynchronize(sync_stop);////////////////////////////////////

      float kernel_time = 0;
      cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
      printf("(9-4-4) Only Heterogeneous kernel execution time: %f ms\n", kernel_time);

      float sync_time1 = 0;
      cudaEventElapsedTime(&sync_time1, kernel_start, sync_start);
      printf("(9-4-4) Only Heterogeneous kernel start ~ sync start: %f ms\n", sync_time1);


      //float kernel_sync_time1 = 0 ; 
      //cudaEventElapsedTime(&kernel_sync_time1, kernel_start, sync_stop);
      //printf("(9-2) kernel start ~ sync stop: %f ms\n", kernel_sync_time1);



      // CUDA 이벤트 제거
      cudaEventDestroy(kernel_start);
      cudaEventDestroy(kernel_stop);
      cudaEventDestroy(sync_start);
      //cudaEventDestroy(sync_stop);
 
      
//----------------------------------------------------------------------------------------------------------------------------      
/* H2D */     
      cudaEvent_t start_h2d, stop_h2d;
      cudaEventCreate(&start_h2d);
      cudaEventCreate(&stop_h2d);
      cudaEventRecord(start_h2d);

      cudaMemcpy((*assignments)[devi].get(), assignment_yy, h_samples_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

      cudaDeviceSynchronize();
      cudaEventRecord(stop_h2d);
      cudaEventSynchronize(stop_h2d);
      float milliseconds_h2d = 0;
      cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);
      printf("(9-4-5) Host To Device Data Transfer Time: %f ms\n", milliseconds_h2d);
      //float total_time = milliseconds6 + milliseconds7 + milliseconds8 + milliseconds_d2h + sync_time1 + milliseconds_h2d;
      //printf("(9-4-6) Total Time: %f ms\n", total_time);
      cudaEventDestroy(start_h2d);
      cudaEventDestroy(stop_h2d);

//----------------------------------------------------------------------------------------------------------------------------------------------------------------  
      /* Retrieve d_changed_number from GPU */
      uint32_t h_changed_number_gpu;
      CUCH(cudaMemcpyFromSymbol(&h_changed_number_gpu, d_changed_number, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost), kmcudaMemoryCopyError); // device -> host
      //printf("(9-2-4) (GPU) Changed Number: %u\n", h_changed_number_gpu);
      //cudaDeviceSynchronize();
      /* Combine CPU and GPU changed numbers */
      uint32_t total_changed_number = h_changed_number_cpu + h_changed_number_gpu;
      printf("(9-4-6) Iteration %d Total Reassignments (CPU + GPU): %u\n", iter, total_changed_number );
      CUCH(cudaMemcpyToSymbol(d_changed_number, &total_changed_number, sizeof(uint32_t), 0, cudaMemcpyHostToDevice), kmcudaMemoryCopyError); // reflect next iteration
//-------------------------------------------------------------------------------------------------------------------------
      prev_execution_time = sync_time1;
      prev_skip_ratio = *current_skip_ratio;

      cudaFreeHost(cpu_sum);
      cudaFreeHost(gpu_sum);

      cudaFreeHost(mark_cpu_yy);
      cudaFreeHost(mark_gpu_yy);
      cudaFreeHost(mark_cpu_number);
      cudaFreeHost(mark_gpu_number);

     // free(h_samples);
      free(h_centroids);
      free(h_assignments_yy);
      free(h_drifts_yy);
      //free(h_passed_yy);


    

    }


//---------------------------------------------------------------------------------------------------------------------------------------------------------------- 

  /* outside else */
  /*  local filter2 end */
  cudaFreeHost(second_min_dists_yy);
  cudaFreeHost(mark_threads_yy);
  cudaFreeHost(calculate_data_point_yy);
  //cudaFree(d_calculate_centroid);

  cudaFreeHost(current_skip_ratio);
  cudaFreeHost(partition_threshold);

  cudaFreeHost(calculate_sum);


  // std::cout << "End capturing power metrics..." << std::endl;   
  // capture_power_metrics_in_interval_end(iter);



  );
  //--------------------------------------------FOR_EACH_DEVI--------------------------------------



























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
  cudaFreeHost(bounds_yy);
  cudaFreeHost(assignment_yy);

  free(h_samples);
 // free(h_centroids);

  //cudaFreeHost(samples_yy);
  //cudaFreeHost(centroids_yyy);
  //cudaFreeHost(assignments_yyy);
  //cudaFreeHost(drifts_yyy);
  

}
//------------------------------------------------------kmeans_cuda_yy------------------------------------------------------------



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