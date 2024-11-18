#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "cpu_function.h"
#include "gpu_function.cuh"

int main() {
    int num_clusters = 18;
    int num_data_points = 6;

    // Create distance array
    float* h_distance;
    cudaHostAlloc((void**)&h_distance, num_clusters * num_data_points * sizeof(float), cudaHostAllocMapped);
    memset(h_distance, 0, num_clusters * num_data_points * sizeof(float));

    float* d_distance;
    cudaHostGetDevicePointer(&d_distance, h_distance, 0);

    // initialize skip array (store the centroids to be skipped)
    int* h_skip;
    cudaHostAlloc((void**)&h_skip, num_clusters * num_data_points * sizeof(int), cudaHostAllocMapped);
    init_skip(h_skip, num_clusters, num_data_points);
    printf("\n=== Clusters to calculate ===\n");
    print_clusters_to_calculate(h_skip, num_clusters, num_data_points);

    int* d_skip;
    cudaHostGetDevicePointer(&d_skip, h_skip, 0);

    // Creat bitmask array
    int bitmask_col = (num_clusters + BITMASK_SIZE - 1) / BITMASK_SIZE;
    BITMASK* h_bitmask;
    cudaHostAlloc((void**)&h_bitmask, num_data_points * bitmask_col * sizeof(BITMASK), cudaHostAllocMapped);

    BITMASK* d_bitmask;
    cudaHostGetDevicePointer(&d_bitmask, h_bitmask, 0);

    // Marking
    marking<<<1, 512>>>(d_bitmask, d_skip, num_clusters, num_data_points);
    cudaDeviceSynchronize();
    printf("\n=== Bitmask (LSB to MSB) ===\n");
    print_bitmask(h_bitmask, num_clusters, num_data_points);

    // Distance calculation
    printf("\n=== Distance Calculation Start ===\n");
    calculate_distance<<<1, 512>>>(d_distance, d_bitmask, num_clusters, num_data_points);
    cudaDeviceSynchronize();


    // Distance calculation results
    printf("\n=== Distance Calculation Results ===\n");
    print_distance(h_distance, num_clusters, num_data_points);

    cudaFreeHost(d_bitmask);
    cudaFreeHost(d_skip);

    return 0;
}
