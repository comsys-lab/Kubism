/********************************************
*
*Kubism - CPU Function
*Implemented by: Seondeok Kim
*
********************************************/

#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

#include <cstdint>
#include <atomic>



template <typename F>
float distance_t (
    const float *v1,
    const float *v2,
    uint64_t v1_size,
    uint16_t features_size,
    uint64_t v1_index);


template <typename F>
void kubism_CPU_WB_HETD(
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
    uint32_t& changed_number_cpu,
    int32_t *mark_threads,
    uint16_t *calculate_data_point,
    uint32_t *partition_threshold
);


template <typename F>
void kubism_CPU_HETD(
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
    uint32_t& changed_number_cpu
);



#endif

