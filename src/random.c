/********************************************
*
*Read Input File 
*Implemented by: Seondeok Kim
*
********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <kmcuda.h>
#include <errno.h> 

/* Read csv input file */
int read_csv(const char *filename, float **data, uint32_t *samples_size, uint16_t *features_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("File opening failed");
        return -1;
    }

    char line[50000];
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Error reading file\n");
        fclose(file);
        return -1;
    }

    *features_size = 0;
    char *token = strtok(line, ",");
    while (token) {
        (*features_size)++;
        token = strtok(NULL, ",");
    }

    fseek(file, 0, SEEK_SET);

    *samples_size = 0;
    while (fgets(line, sizeof(line), file)) {
        (*samples_size)++;
    }

    fseek(file, 0, SEEK_SET);

    *data = (float *)malloc((*samples_size) * (*features_size) * sizeof(float));
    if (!*data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return -1;
    }

    uint32_t row = 0;
    while (fgets(line, sizeof(line), file) && row < *samples_size) {
        token = strtok(line, ",");
        uint32_t col;
        for (col = 0; col < *features_size; col++) {
            if (token) {
                char *endptr;
                errno = 0;
                (*data)[row * (*features_size) + col] = strtof(token, &endptr);
                if (errno != 0 || endptr == token) {
                    fprintf(stderr, "Invalid data at row %u, col %u; filling with 0.0\n", row, col);
                    (*data)[row * (*features_size) + col] = 0.0; // Default value for invalid data
                }
                token = strtok(NULL, ",");
            } else {
                fprintf(stderr, "Missing data at row %u, col %u; filling with 0.0\n", row, col);
                (*data)[row * (*features_size) + col] = 0.0; // Default value for missing data
            }
        }
        row++;
    }

    fclose(file);
    return 0;
}

int main(int argc, const char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <csv file>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    uint32_t samples_size;
    uint16_t features_size;
    float *samples;

    if (read_csv(filename, &samples, &samples_size, &features_size) != 0) {
        return 1;
    }

    const int clusters_size = 1024;

    size_t samples_memory_size = samples_size * features_size * sizeof(float);
    size_t centroids_memory_size = clusters_size * features_size * sizeof(float);
    size_t assignments_memory_size = samples_size * sizeof(uint32_t);

    float *centroids = (float*)malloc(centroids_memory_size);
    if (!centroids) {
        fprintf(stderr, "Memory allocation for centroids failed\n");
        free(samples);
        return 1;
    }

    uint32_t *assignments = (uint32_t*)malloc(assignments_memory_size);
    if (!assignments) {
        fprintf(stderr, "Memory allocation for assignments failed\n");
        free(samples);
        free(centroids);
        return 1;
    }

    float average_distance;

    

    /* K-Means Configuration */
    KMCUDAResult result = kmeans_cuda(
        kmcudaInitMethodRandom, NULL,
        0.01,
        0.1,
        kmcudaDistanceMetricL2,
        samples_size, features_size, clusters_size,
        0xDEADBEEF,
        1,
        -1,
        0,
        2,
        samples, centroids, assignments, &average_distance);

    if (result != kmcudaSuccess) {
        fprintf(stderr, "KMeans clustering failed with error code %d\n", result);
        free(samples);
        free(centroids);
        free(assignments);
        return 1;
    }

    free(samples);
    free(centroids);
    free(assignments);

    return 0;
}
