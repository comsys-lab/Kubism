#include <stdio.h>
#include "gpu_function.cuh"

__global__
void marking(BITMASK* bitmask, int* skip, int num_clusters, int num_data_points) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_data_points) {
        return;
    }

    int bitmask_col = (num_clusters + BITMASK_SIZE - 1) / BITMASK_SIZE;

    register BITMASK mask = 0;
    for (int c = 0; c < num_clusters; c++) {
        // register BITMASK is full
        if (c % BITMASK_SIZE == 0 && c != 0) {
            bitmask[tid * bitmask_col + c / BITMASK_SIZE - 1] = mask;
            mask = 0;
        }

        // Check if the centroid is skipped
        if (skip[tid * num_clusters + c] == 0)
            continue;

        mask |= (1 << (c % BITMASK_SIZE));
    }

    bitmask[tid * bitmask_col + bitmask_col - 1] = mask;

    // for (int c = 0; c < bitmask_col; c++) {
    //     BITMASK mask = 0;
    //     for (int i = 0; i < BITMASK_SIZE; i++) {
    //         int c_idx = c * BITMASK_SIZE + i;
    //         if (skip[tid * num_clusters + c_idx]) {
    //             mask |= (1 << i);
    //         }
    //     }

    //     bitmask[tid * bitmask_col + c] = mask;
    // }

    // printf("%d %d\n", bitmask[tid * bitmask_col], bitmask[tid * bitmask_col + 1]);
}

__global__
void calculate_distance(float* d_distance, BITMASK* bitmask, int num_clusters, int num_data_points) {
    int tid = threadIdx.x;
    
    if (tid >= num_data_points) {
        return;
    }

    int bitmask_col = (num_clusters + BITMASK_SIZE - 1) / BITMASK_SIZE;

    int bitmask_idx = 0;
    register BITMASK mask = bitmask[tid * bitmask_col + bitmask_idx];

    // Calculate distance
    for (int i = 0; i < num_clusters; i++) {
        // Find a non-zero bitmask
        while (mask == 0 && bitmask_idx < bitmask_col - 1) {
            bitmask_idx++;
            mask = bitmask[tid * bitmask_col + bitmask_idx];
        }

        // If mask is 0 and bitmask_idx is the last one, terminate distance calculation.
        if (bitmask_idx == bitmask_col - 1 && mask == 0)
            break;

        // Find the index of the centroid for distance calculation.
        int first_set_idx = __ffs(mask) - 1;
        int centroid_idx = bitmask_idx * BITMASK_SIZE + first_set_idx;

        // Distance calculation code
        float distance = cal_dist(i, tid, centroid_idx, 100000000);
        
        // Remove the processed centroid from the bitmask
        mask &= ~(1 << first_set_idx);

        // Store the results
        d_distance[tid * num_clusters + centroid_idx] = distance;
    }
}

__device__
float cal_dist (int iteration, int tid, int centroid_idx, int dummy)
{
    printf("[Iteration %d] tid %2d calculates distance with centroid %2d\n", iteration, tid, centroid_idx);

    float distance = 0.;

    // Delay...
    for (int i = 0; i < dummy; i++) {
        distance++;
    }

    if (distance > 0) {
        distance = float(iteration);
    }

    return distance + 1.f;
}