#ifndef __GPU_FUNCTION_CUH__
#define __GPU_FUNCTION_CUH__

#include "common.h"

__global__ void calculate_distance(float* d_distance, BITMASK* bitmask, int num_cluster, int num_data_points);
__global__ void marking(BITMASK* bitmask, int* skip, int num_clusters, int num_data_points);
__global__ void calculate_distance(float* d_distance, BITMASK* bitmask, int num_cluster, int num_data_points);
__device__ float cal_dist (int iteration, int tid, int centroid_idx, int dummy);

#endif