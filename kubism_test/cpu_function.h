#include <stdlib.h>
#include "common.h"

void init_skip(int* skip, int num_clusters, int num_data_points);
void print_clusters_to_calculate(int* skip, int num_clusters, int num_data_points);
void print_bitmask(BITMASK* bitmask, int num_cluster, int num_data_points);
void print_distance(float* distance, int num_clusters, int num_data_points);