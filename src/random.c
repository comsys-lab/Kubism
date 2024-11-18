// #include <assert.h>
// #include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <kmcuda.h>

// #define MAX_FEATURES 29 // feature size를 128로 제한

// // CSV 파일에서 데이터를 읽어오는 함수
// int read_csv(const char *filename, float **data, uint32_t *samples_size, uint16_t *features_size) {
//     FILE *file = fopen(filename, "r");
//     if (!file) {
//         perror("File opening failed");
//         return -1;
//     }

//     char line[2000000];  // 버퍼 크기를 300000으로 설정

//     // 첫 번째 줄을 읽어 feature 개수 확인
//     if (!fgets(line, sizeof(line), file)) {
//         fprintf(stderr, "Error reading file\n");
//         fclose(file);
//         return -1;
//     }

//     // features_size를 128로 제한
//     *features_size = 0;
//     char *token = strtok(line, ",");
//     while (token && *features_size < MAX_FEATURES) {
//         (*features_size)++;
//         token = strtok(NULL, ",");
//     }

//     // 파일을 처음으로 되돌림
//     fseek(file, 0, SEEK_SET);

//     // 샘플 수를 계산
//     *samples_size = 0;
//     while (fgets(line, sizeof(line), file)) {
//         (*samples_size)++;
//     }

//     // 파일을 다시 처음으로 이동
//     fseek(file, 0, SEEK_SET);

//     // 데이터를 저장할 메모리 할당
//     *data = (float *)malloc((*samples_size) * MAX_FEATURES * sizeof(float));
//     if (!*data) {
//         fprintf(stderr, "Memory allocation failed\n");
//         fclose(file);
//         return -1;
//     }

//     // CSV 데이터를 float 배열로 변환, 첫 128개 열만 읽음
//     uint32_t row = 0;
//     while (fgets(line, sizeof(line), file) && row < *samples_size) {
//         token = strtok(line, ",");
//         for (uint32_t col = 0; col < MAX_FEATURES; col++) {
//             if (token) {
//                 // 작은 따옴표를 제거
//                 if (token[0] == '\'') {
//                     memmove(token, token + 1, strlen(token));
//                 }

//                 // 문자열을 float로 변환
//                 (*data)[row * MAX_FEATURES + col] = strtof(token, NULL);
//                 token = strtok(NULL, ",");
//             } else {
//                 fprintf(stderr, "Error reading file at row %u, col %u\n", row, col);
//                 free(*data);
//                 fclose(file);
//                 return -1;
//             }
//         }
//         row++;
//     }

//     fclose(file);
//     return 0;
// }

// int main(int argc, const char **argv) {
//     if (argc != 2) {
//         fprintf(stderr, "Usage: %s <csv file>\n", argv[0]);
//         return 1;
//     }

//     const char *filename = argv[1];
//     uint32_t samples_size;
//     uint16_t features_size = MAX_FEATURES;
//     float *samples;

//     // CSV 파일 읽기
//     if (read_csv(filename, &samples, &samples_size, &features_size) != 0) {
//         return 1;
//     }

//     const int clusters_size = 1024;  // # of centroids

//     size_t samples_memory_size = samples_size * MAX_FEATURES * sizeof(float);
//     size_t centroids_memory_size = clusters_size * MAX_FEATURES * sizeof(float);
//     size_t assignments_memory_size = samples_size * sizeof(uint32_t);

//     // 디버깅용 출력
//     printf("Samples size: %u, Features size: %u, Clusters size: %d\n", samples_size, features_size, clusters_size);
//     printf("Samples memory size: %zu bytes\n", samples_memory_size);
//     printf("Centroids memory size: %zu bytes\n", centroids_memory_size);
//     printf("Assignments memory size: %zu bytes\n", assignments_memory_size);

//     // 메모리 할당
//     float *centroids = (float*)malloc(centroids_memory_size);
//     if (!centroids) {
//         fprintf(stderr, "Memory allocation for centroids failed\n");
//         free(samples);
//         return 1;
//     }

//     uint32_t *assignments = (uint32_t*)malloc(assignments_memory_size);
//     if (!assignments) {
//         fprintf(stderr, "Memory allocation for assignments failed\n");
//         free(samples);
//         free(centroids);
//         return 1;
//     }

//     // KMeans 알고리즘 실행
//     float average_distance;
//     KMCUDAResult result = kmeans_cuda(
//         kmcudaInitMethodRandom, NULL,  // random centroids initialization
//         0.01,                         // tolerance, less than 1% of the samples are reassigned in the end
//         0.1,                           // activate Yinyang refinement with 0.1 threshold
//         kmcudaDistanceMetricL2,        // Euclidean distance
//         samples_size, MAX_FEATURES, clusters_size,
//         0xDEADBEEF,                    // random generator seed
//         1,                             // use all available CUDA devices : 1 - first device, 2 - second device, 3 - first & second device
//         -1,                            // samples are supplied from host
//         0,                             // not in float16x2 mode
//         2,                             // moderate verbosity
//         samples, centroids, assignments, &average_distance);

//     // 결과 확인
//     if (result != kmcudaSuccess) {
//         fprintf(stderr, "KMeans clustering failed with error code %d\n", result);
//         free(samples);
//         free(centroids);
//         free(assignments);
//         return 1;
//     }

//     // 메모리 해제
//     free(samples);
//     free(centroids);
//     free(assignments);

//     return 0;
// }

























// #include <assert.h>
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <ctype.h>    // 추가: isspace 함수 사용을 위해
// #include <kmcuda.h>

// #define MAX_FEATURES 29 // feature size를 29로 제한

// // 문자열의 앞뒤 공백을 제거하는 함수
// char* trim_whitespace(char *str) {
//     char *end;

//     // 앞쪽 공백 제거
//     while(isspace((unsigned char)*str)) str++;

//     if(*str == 0)  // 모든 문자가 공백인 경우
//         return str;

//     // 뒤쪽 공백 제거
//     end = str + strlen(str) - 1;
//     while(end > str && isspace((unsigned char)*end)) end--;

//     // 새로운 종료 문자 설정
//     *(end+1) = 0;

//     return str;
// }

// // CSV 파일에서 데이터를 읽어오는 함수
// int read_csv(const char *filename, float **data, uint32_t *samples_size, uint16_t *features_size) {
//     FILE *file = fopen(filename, "r");
//     if (!file) {
//         perror("File opening failed");
//         return -1;
//     }

//     char line[2000000];  // 버퍼 크기를 2,000,000으로 설정

//     // 첫 번째 줄을 읽어 feature 개수 확인
//     if (!fgets(line, sizeof(line), file)) {
//         fprintf(stderr, "Error reading file\n");
//         fclose(file);
//         return -1;
//     }

//     // features_size를 MAX_FEATURES로 제한하면서 실제 열 개수 확인
//     *features_size = 0;
//     char *token = strtok(line, ",");
//     while (token && *features_size < MAX_FEATURES) {
//         (*features_size)++;
//         token = strtok(NULL, ",");
//     }

//     // 첫 줄의 열 개수가 MAX_FEATURES와 일치하는지 확인
//     if (*features_size != MAX_FEATURES) {
//         fprintf(stderr, "Unexpected number of features in the first row: %u (expected %d)\n", *features_size, MAX_FEATURES);
//         fclose(file);
//         return -1;
//     }

//     // 파일을 처음으로 되돌림
//     fseek(file, 0, SEEK_SET);

//     // 샘플 수를 계산
//     *samples_size = 0;
//     while (fgets(line, sizeof(line), file)) {
//         (*samples_size)++;
//     }

//     // 파일을 다시 처음으로 이동
//     fseek(file, 0, SEEK_SET);

//     // 데이터를 저장할 메모리 할당
//     *data = (float *)malloc((*samples_size) * MAX_FEATURES * sizeof(float));
//     if (!*data) {
//         fprintf(stderr, "Memory allocation failed\n");
//         fclose(file);
//         return -1;
//     }

//     // CSV 데이터를 float 배열로 변환, 첫 features_size개 열만 읽음
//     uint32_t row = 0;
//     uint32_t error_count = 0; // 오류 발생 횟수 카운트
//     while (fgets(line, sizeof(line), file) && row < *samples_size) {
//         token = strtok(line, ",");
//         for (uint32_t col = 0; col < MAX_FEATURES; col++) {
//             if (token) {
//                 // 작은 따옴표를 제거
//                 if (token[0] == '\'') {
//                     memmove(token, token + 1, strlen(token));
//                 }

//                 // 양쪽 공백 제거
//                 token = trim_whitespace(token);

//                 // 문자열을 float로 변환
//                 char *endptr;
//                 float val = strtof(token, &endptr);

//                 // 변환 실패 시 기본값으로 설정하고 경고 출력
//                 if (endptr == token) {
//                     fprintf(stderr, "Warning: Error parsing float at row %u, col %u: '%s'. Setting to 0.0f\n", row+1, col+1, token);
//                     (*data)[row * MAX_FEATURES + col] = 0.0f;
//                     error_count++;
//                 } else {
//                     // 변환 후 남은 문자가 공백인지 확인
//                     while (isspace((unsigned char)*endptr)) endptr++;
//                     if (*endptr != '\0' && *endptr != '\n') {
//                         fprintf(stderr, "Warning: Unexpected characters after float at row %u, col %u: '%s'. Setting to 0.0f\n", row+1, col+1, endptr);
//                         (*data)[row * MAX_FEATURES + col] = 0.0f;
//                         error_count++;
//                     } else {
//                         (*data)[row * MAX_FEATURES + col] = val;
//                     }
//                 }

//                 token = strtok(NULL, ",");
//             } else {
//                 // 누락된 토큰이 있는 경우 기본값으로 설정하고 경고 출력
//                 fprintf(stderr, "Warning: Missing token at row %u, col %u. Setting to 0.0f\n", row+1, col+1);
//                 (*data)[row * MAX_FEATURES + col] = 0.0f;
//                 error_count++;
//             }
//         }
//         row++;
//     }

//     fclose(file);

//     if (error_count > 0) {
//         fprintf(stderr, "Total warnings during CSV parsing: %u\n", error_count);
//     }

//     return 0;
// }

// int main(int argc, const char **argv) {
//     if (argc != 2) {
//         fprintf(stderr, "Usage: %s <csv file>\n", argv[0]);
//         return 1;
//     }

//     const char *filename = argv[1];
//     uint32_t samples_size;
//     uint16_t features_size = MAX_FEATURES;
//     float *samples;

//     // CSV 파일 읽기
//     if (read_csv(filename, &samples, &samples_size, &features_size) != 0) {
//         return 1;
//     }

//     const int clusters_size = 1024;  // # of centroids

//     size_t samples_memory_size = (size_t)samples_size * features_size * sizeof(float);
//     size_t centroids_memory_size = (size_t)clusters_size * features_size * sizeof(float);
//     size_t assignments_memory_size = (size_t)samples_size * sizeof(uint32_t);

//     // 디버깅용 출력
//     printf("Samples size: %u, Features size: %u, Clusters size: %d\n", samples_size, features_size, clusters_size);
//     printf("Samples memory size: %zu bytes\n", samples_memory_size);
//     printf("Centroids memory size: %zu bytes\n", centroids_memory_size);
//     printf("Assignments memory size: %zu bytes\n", assignments_memory_size);

//     // 메모리 할당
//     float *centroids = (float*)malloc(centroids_memory_size);
//     if (!centroids) {
//         fprintf(stderr, "Memory allocation for centroids failed\n");
//         free(samples);
//         return 1;
//     }

//     uint32_t *assignments = (uint32_t*)malloc(assignments_memory_size);
//     if (!assignments) {
//         fprintf(stderr, "Memory allocation for assignments failed\n");
//         free(samples);
//         free(centroids);
//         return 1;
//     }

//     // KMeans 알고리즘 실행
//     float average_distance;
//     KMCUDAResult result = kmeans_cuda(
//         kmcudaInitMethodRandom, NULL,  // random centroids initialization
//         0.01,                           // tolerance, less than 1% of the samples are reassigned in the end
//         0.1,                            // activate Yinyang refinement with 0.1 threshold
//         kmcudaDistanceMetricL2,         // Euclidean distance
//         samples_size, features_size, clusters_size,
//         0xDEADBEEF,                     // random generator seed
//         1,                              // use all available CUDA devices : 1 - first device, 2 - second device, 3 - first & second device
//         -1,                             // samples are supplied from host
//         0,                              // not in float16x2 mode
//         2,                              // moderate verbosity
//         samples, centroids, assignments, &average_distance);

//     // 결과 확인
//     if (result != kmcudaSuccess) {
//         fprintf(stderr, "KMeans clustering failed with error code %d\n", result);
//         free(samples);
//         free(centroids);
//         free(assignments);
//         return 1;
//     }

//     // 결과 출력 (옵션)
//     /*
//     for(uint32_t i = 0; i < samples_size; i++) {
//         printf("Sample %u: Cluster %u\n", i, assignments[i]);
//     }
//     */

//     // 메모리 해제
//     free(samples);
//     free(centroids);
//     free(assignments);

//     return 0;
// }

int read_csv(const char *filename, float **data, uint32_t *samples_size, uint16_t *features_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("File opening failed");
        return -1;
    }
 
   
    char line[50000];  // Increase the buffer size
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
        for (uint32_t col = 0; col < *features_size; col++) {
            if (token) {
                (*data)[row * (*features_size) + col] = strtof(token, NULL);
                token = strtok(NULL, ",");
            } else {
                fprintf(stderr, "Error reading file at row %u, col %u\n", row, col);
                free(*data);
                fclose(file);
                return -1;
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
 
    // CSV file read
    if (read_csv(filename, &samples, &samples_size, &features_size) != 0) {
        return 1;
    }
 
    const int clusters_size = 1024; // # of centroids
 
   
    size_t samples_memory_size = samples_size * features_size * sizeof(float);
    size_t centroids_memory_size = clusters_size * features_size * sizeof(float);
    size_t assignments_memory_size = samples_size * sizeof(uint32_t);
 
    // printf("-------------------------------------------------------------------------------------------------------------\n");
    // printf("Samples memory size: %zu bytes\n", samples_memory_size);
    // printf("Centroids memory size: %zu bytes\n", centroids_memory_size);
    // printf("Assignments memory size: %zu bytes\n", assignments_memory_size);
    // printf("Total memory size: %zu bytes\n", samples_memory_size + centroids_memory_size + assignments_memory_size);
    // printf("--------------------------------------------------------------------------------------------------------------\n");
    // printf("\n");
    // initialization for clustering
    float *centroids = (float*)malloc(centroids_memory_size);
    if (!centroids) {
        fprintf(stderr, "Memory allocation for centroids failed\n");
        free(samples);
        return 1;
    }
    //printf("Memory allocation for centroids successful\n");
 
    uint32_t *assignments = (uint32_t*)malloc(assignments_memory_size);
    if (!assignments) {
        fprintf(stderr, "Memory allocation for assignments failed\n");
        free(samples);
        free(centroids);
        return 1;
    }
    //printf("Memory allocation for assignments successful\n");
 
 
 
    // Execute KMeans algorithm
    float average_distance;
    KMCUDAResult result = kmeans_cuda(
        kmcudaInitMethodRandom, NULL,  // random centroids initialization
        0.01,                            // tolerance, less than 1% of the samples are reassigned in the end
        0.1,                             // activate Yinyang refinement with 0.1 threshold
        kmcudaDistanceMetricL2,          // Euclidean distance
        samples_size, features_size, clusters_size,
        0xDEADBEEF,                      // random generator seed
        1,                               // use all available CUDA devices : 1 - first device, 2 - second device, 3 - first & second device
        -1,                              // samples are supplied from host
        0,                               // not in float16x2 mode
        2,                               // moderate verbosity
        samples, centroids, assignments, &average_distance);
 
 
 
    // result
    if (result != kmcudaSuccess) {
        fprintf(stderr, "KMeans clustering failed with error code %d\n", result);
        free(samples);
        free(centroids);
        free(assignments);
        return 1;
    }
 
 
    // free memory
    free(samples);
    free(centroids);
    free(assignments);
 
    return 0;
}