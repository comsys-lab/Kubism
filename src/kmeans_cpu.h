#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

#include <cstdint>
#include <atomic>
//#include "metric_abstraction.h"



template <typename F>
float distance_t (
    const float *v1,
    const float *v2,
    uint64_t v1_size,
    uint16_t features_size,
    uint64_t v1_index);


template <typename F>
void local_filter_cpu1(
    //const uint32_t offset, 
    //const uint32_t length,
    const uint32_t samples_size,
    const uint32_t clusters_size,
    const uint16_t features_size,
    const uint32_t yy_groups_size, 
    const float *samples,
    const uint32_t *mark_cpu, 
    const float *centroids,
    const uint32_t *groups, 
    const float *drifts,
    uint32_t *assignments, 
    float *bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    //std::atomic<uint32_t>& changed_number_cpu
    uint32_t& changed_number_cpu,
   // uint32_t *calculate_centroid_cpu
    int32_t *mark_threads,
    uint16_t *calculate_data_point,
    uint32_t *partition_threshold
   // uint32_t *passed,
   // uint32_t global_to_local
);


template <typename F>
void local_filter_cpu2(
    //const uint32_t offset, 
    //const uint32_t length,
    const uint32_t samples_size,
    const uint32_t clusters_size,
    const uint16_t features_size,
    const uint32_t yy_groups_size, 
    const float *samples,
    //udevptrs<float>&samples,
    const uint32_t *mark_cpu, 
    const float *centroids,
    const uint32_t *groups, 
    const float *drifts,
    uint32_t *assignments, 
    float *bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    //std::atomic<uint32_t>& changed_number_cpu
    uint32_t& changed_number_cpu
   // uint32_t *calculate_centroid_cpu
    

);



#endif