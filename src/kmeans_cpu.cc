#include "kmeans_cpu.h"
#include <omp.h>
#include <vector>
#include <thread>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <atomic>
#include <cfloat>
//#include "metric_abstraction.h"




/*float*/
// template <typename F>
// inline float distance_t (
//     const float *v1,
//     const float *v2,
//     uint64_t v1_size,
//     uint16_t features_size,
//     uint64_t v1_index){

//     double dist = 0.0;
//     double corr = 0.0;

//     for(uint64_t f = 0 ; f < features_size ; f++){
//         double d = v1[v1_size * f + v1_index] -  v2[f];
//         //double y = corr * d + d * d;
//         double y = std::fma(corr, d, d * d);
//         double t = dist + y;

//         corr = y - (t - dist);
//         dist = t;
//     }
//     return std::sqrt(dist);
// }


/*double*/
// template <typename F>
// inline float distance_t (
//     const float *v1,
//     const float *v2,
//     uint64_t v1_size,
//     uint16_t features_size,
//     uint64_t v1_index){

//     float dist = 0.0f;
//     float corr = 0.0f;

//     for(uint64_t f = 0 ; f < features_size ; f++){
//         float d = v1[v1_size * f + v1_index] -  v2[f];
//         //double y = corr * d + d * d;
//         float y = std::fma(corr, d, d * d);
//         float t = dist + y;

//         corr = y - (t - dist);
//         dist = t;
//     }
//     return std::sqrt(dist);
// }

template <typename F>
inline float distance_t(
    const float *v1,
    const float *v2,
    uint32_t v1_size,
    uint16_t features_size,
    uint32_t v1_index) {

    float dist = 0.0f;

    for (uint16_t f = 0; f < features_size; f++) {
        float d = v1[v1_size * f + v1_index] - v2[f];
        dist += d * d;
    }
    return std::sqrt(dist);
}


template <typename F>
void local_filter_cpu1(
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
    //int32_t *passed,
    //uint32_t global_to_local
    ) {

    omp_set_num_threads(omp_get_max_threads());
//-------------------------------------------------------------------------------------------------------
    changed_number_cpu = 0 ;
    //uint32_t distribution_threshold = passed[0] + *partition_threshold;
    #pragma omp parallel for
    //for (uint32_t i = 0; i < distribution_threshold; i++) {
    for (uint32_t i = 0; i < *mark_cpu_number; i++) {
        uint32_t sample = mark_cpu[i];
        //uint32_t sample = passed[i];
        float upper_bound = bounds[sample];
        uint32_t cluster = assignments[sample];
        uint32_t nearest = cluster;
        float min_dist = upper_bound;
        float second_min_dist = second_min_dists[sample];

        for (uint32_t i = 0; i < 32; i++) {
            int32_t mask = mark_threads[sample * 32 + i];

            if(mask == 0 && i < 31){
                uint32_t new_mask = 0;
                for(uint32_t j = 1 ; j < (32 - i ); j++){
                    new_mask = mark_threads[sample * 32 + i + j];
                    if(new_mask != 0){
                        i += j;
                        mask = new_mask;
                        break;
                    }

                }
            }

            while (mask != 0) {
                int first_bit_pos = __builtin_ffs(mask) - 1;  
                uint32_t c = i * 32 + first_bit_pos; 

                
                float dist = 0;
                for (uint16_t f = 0; f < features_size; f++) {
                    float d = samples[samples_size * f + sample] - centroids[c * features_size + f];
                    dist += d * d;
                }
                dist = sqrt(dist);

                if (dist < min_dist) {
                    second_min_dist = min_dist;
                    min_dist = dist;
                    nearest = c;
                } else if (dist < second_min_dist) {
                    second_min_dist = dist;
                }
                mask &= ~(1 << first_bit_pos);
            }
        }

        uint32_t nearest_group = groups[nearest];
        uint32_t previous_group = groups[cluster];
        bounds[samples_size * (1 + nearest_group) + sample] = second_min_dist;

        if (nearest_group != previous_group) {
            uint64_t gindex = static_cast<uint64_t>(samples_size) * (1 + previous_group) + sample;
            float pb = bounds[gindex];
            if (pb > upper_bound) {
                bounds[gindex] = upper_bound;
            }
        }
        bounds[sample] = min_dist;

        if (cluster != nearest) {
            assignments[sample] = nearest;
            #pragma omp atomic
            changed_number_cpu += 1;
        }
    }
}






















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
    //uint32_t *calculate_centroid_cpu
   



    ){

    omp_set_num_threads(omp_get_max_threads());
//-------------------------------------------------------------------------------------------------------
    changed_number_cpu = 0 ;
    
    //uint32_t distribution_threshold = passed[0] + *partition_threshold;
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < *mark_cpu_number; i++) {
    //for (uint32_t i = 0; i < distribution_threshold ; i++) {
       // uint32_t sample = passed[i];
        uint32_t sample = mark_cpu[i];
        // if (sample >= *partition_threshold) {
        //     continue;
        // }
        float upper_bound = bounds[sample];
        uint32_t cluster = assignments[sample];
        uint32_t nearest = cluster;
        float min_dist = upper_bound;
        float second_min_dist = second_min_dists[sample];
        uint32_t doffset = clusters_size * features_size;
 

        // Process each cluster
        for (uint32_t c = 0; c < clusters_size; c++) {
            if (c == cluster) {
                //printf("CPU sample = %u skip with centroid = %u cluster = %u\n", sample, c, cluster);
                continue;
            }
            uint32_t group = groups[c];
            if (group >= yy_groups_size){ 
                //printf("CPU sample = %u skip with centroid = %u group = %u\n", sample, c, group);
                continue;
            }

            float lower_bound = bounds[static_cast<uint64_t>(samples_size) * (1 + group) + sample];
            
            // Local Filtering #1
            if (lower_bound >= upper_bound) {
                if (lower_bound < second_min_dist) {
                    second_min_dist = lower_bound;
                }
                //printf("CPU sample = %u skip with centroid = %u group = %u upper bound = %f lower bound = %f \n", sample, c, group, upper_bound, lower_bound);
                continue;
            }
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------            

            lower_bound += drifts[group] - drifts[doffset + c];
            //lower_bound += drift_a - drift_b;

            //printf("CPU group = %u doffset = %u c = %u drifts[group] = %f drifts[doffset+c] = %f lower bound = %f \n", group, doffset, c, drifts[group], drifts[doffset+c], lower_bound);
            // Local Filtering #2
            if (lower_bound > second_min_dist) {
                //printf("CPU sample = %u skip with centroid = %u group = %u second min dist = %f lower bound = %f \n", sample, c, group, second_min_dist, lower_bound);
                continue;
            }
            

            // Calculate distance
            float dist = distance_t<float>(samples, centroids + c * features_size, samples_size, features_size, sample);
            // #pragma omp atomic
            // calculate_centroid_cpu[c] += 1;
            //printf("CPU sample = %u skip with centroid = %u dist = %f\n", sample, c, dist);
            // Update nearest cluster and distances
            if (dist < min_dist) {
                second_min_dist = min_dist;
                min_dist = dist;
                nearest = c;
            } else if (dist < second_min_dist) {
                second_min_dist = dist;
            }
            
        }

        uint32_t nearest_group = groups[nearest];
        uint32_t previous_group = groups[cluster];
        bounds[static_cast<uint64_t>(samples_size) * (1 + nearest_group) + sample] = second_min_dist;
        if (nearest_group != previous_group) {
            uint64_t gindex = static_cast<uint64_t>(samples_size) * (1 + previous_group) + sample;
            float pb = bounds[gindex];
            if (pb > upper_bound) {
                bounds[gindex] = upper_bound;
            }
        }
        bounds[sample] = min_dist;

        if (cluster != nearest) {
            assignments[sample] = nearest;
            //changed_number_cpu.fetch_add(1, std::memory_order_relaxed);
            #pragma omp atomic
            changed_number_cpu += 1;
        }
    }
    #pragma omp barrier
//     end_time = omp_get_wtime();

//    printf("(9-2) CPU Function Execution time: %f milliseconds\n", (end_time - start_time) * 1000);
}





//template void local_filter_cpu<float>(const float*, const uint32_t*, const float*, const uint32_t*, const float*, uint32_t*, float*, uint32_t);

template void local_filter_cpu1<float>(
    const uint32_t samples_size,
    const uint32_t clusters_size,
    const uint16_t features_size,
    const uint32_t yy_groups_size,
    const float* samples,
    const uint32_t *mark_cpu, 
    const float* centroids,
    const uint32_t* groups, 
    const float* drifts,
    uint32_t* assignments, 
    float* bounds,
    float *second_min_dists,
    uint32_t *mark_cpu_number,
    //std::atomic<uint32_t>& changed_number // Ensure this matches the declaration
    uint32_t& changed_number_cpu,
   // uint32_t *calculate_centroid_cpu
    int32_t *mark_threads,
    uint16_t *calculate_data_point,
    uint32_t *partition_threshold
    //uint32_t *passed,
    //uint32_t global_to_local

);




template void local_filter_cpu2<float>(
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
    //uint32_t *calculate_centroid_cpu
    

);