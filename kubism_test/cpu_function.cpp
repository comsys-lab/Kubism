#include <stdio.h>
#include "cpu_function.h"

void init_skip(int* skip, int num_clusters, int num_data_points) 
{
    // Initialize skip array
    for (int i = 0; i < num_clusters; i++) {
        // Thread 0: 0 2 4 6 8 10 ...
        if (i % 2 == 0) {
            skip[i] = 1;
        }
        else {
            skip[i] = 0;
        }

        // Thread 1: 1 5 9 13 17 ...
        if (i % 4 == 1) {
            skip[num_clusters + i] = 1;
        }
        else {
            skip[num_clusters + i] = 0;
        }

        // Thread 2: Only the last centroid
        if (i == num_clusters - 1) {
            skip[2 * num_clusters + i] = 1;
        }
        else {
            skip[2 * num_clusters + i] = 0;
        }

        // Thread 3: No centroid
        skip[3 * num_clusters + i] = 0;

        // Thread 4: Only the first centroid
        if (i == 0) {
            skip[4 * num_clusters + i] = 1;
        }
        else {
            skip[4 * num_clusters + i] = 0;
        }

        // Thread 5~:
        for (int j = 5; j < num_data_points; j++) {
            skip[j * num_clusters + i] = rand() % 2;
        }
    }
}

void print_clusters_to_calculate(int* skip, int num_clusters, int num_data_points)
{
    printf("Cluster     : ");
    for (int i = 0; i < num_clusters; i++) {
        printf("%2d ", i);
        if (i % BITMASK_SIZE == BITMASK_SIZE - 1) {
            printf(" ");
        }
    }
    printf("\n");
    for (int i = 0; i < num_data_points; i++) {
        printf("Data point %d: ", i);
        for (int j = 0; j < num_clusters; j++) {
            printf(" %s ", skip[i * num_clusters + j] ? "O" : ".");
            if (j % BITMASK_SIZE == BITMASK_SIZE - 1) {
                printf(" ");
            }
        }
        printf("\n");
    }
}

void print_bitmask(BITMASK* bitmask, int num_clusters, int num_data_points)
{
    int bitmask_col = (num_clusters + BITMASK_SIZE - 1) / BITMASK_SIZE;
    
    for (int i = 0; i < num_data_points; i++) {
        printf("Data point %d: ", i);
        for (int j = 0; j < bitmask_col; j++) {
            for (int k = 0; k < 8; k++) {
                printf("%d", bitmask[i * bitmask_col + j] & (1 << k) ? 1 : 0);
            }
            printf(" ");
        }
        printf("\n");
    }
}

void print_distance(float* distance, int num_clusters, int num_data_points)
{
    printf("Cluster     : ");
    for (int i = 0; i < num_clusters; i++) {
        printf("%2d ", i);
        if (i % BITMASK_SIZE == BITMASK_SIZE - 1) {
            printf(" ");
        }
    }
    printf("\n");
    for (int i = 0; i < num_data_points; i++) {
        printf("Data point %d: ", i);
        for (int j = 0; j < num_clusters; j++) {
            if (distance[i * num_clusters + j] > 0) {
                printf("%2d ", (int) distance[i * num_clusters + j]);
            }
            else {
                printf(" . ");
            }
            if (j % BITMASK_SIZE == BITMASK_SIZE - 1) {
                printf(" ");
            }
        }
        printf("\n");
    }
}