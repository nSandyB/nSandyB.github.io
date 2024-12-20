#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include<cuda_runtime.h>

#define SIZE 10000000  // size of the array
#define BLOCK_SIZE 256 

void serial_add( int *a, int *b, int *result, int size){
    for(int i =0; i<size; i++){
        result[i]= a[i]+b[i];
    }
}

void parallel_add(int *a, int *b, int *result, int size){
    #pragma omp parallel for
    for(int i=0; i<size; i++){
        result[i] = a[i]+b[i];
    } 
}

__global__ void vector_add(const int *a, const int *b, int *result, int size){
    int index = blockDim.x*blockIdx.x +threadIdx.x;
    if(index<size){
        result[index]= a[index]+b[index];
    }
}

int main(){
    int *a = new int[SIZE];
    int * b = new int[SIZE];
    int * serial_result =  new int[SIZE];
    int *parallel_result = new int[SIZE];
    int *host_result = new int[SIZE];

    for(int i=0; i<SIZE; i++){
        a[i]= rand()%100;
        b[i]= rand()%100;
    }

    int *d_a, *d_b, *d_result;

    cudaMalloc(&d_a, SIZE*sizeof(int));
    cudaMalloc(&d_b, SIZE*sizeof(int));
    cudaMalloc(&d_result, SIZE*sizeof(int));

    cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);




    clock_t start_serial = clock();
    serial_add(a, b, serial_result, SIZE);
    clock_t stop_serial = clock();
    double time_serial = (double(stop_serial- start_serial))/ CLOCKS_PER_SEC;

    double start_para = omp_get_wtime();
    parallel_add(a, b, parallel_result, SIZE);
    double stop_para = omp_get_wtime();
    double time_para = stop_para-start_para;

    int grid_size = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Calculate number of blocks
    clock_t start_cudaParallel = clock();
    vector_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, SIZE);
    cudaDeviceSynchronize();  // Ensure all threads are done
    clock_t end_cudaParallel = clock();
    double time_cudaParallel = ((double)(end_cudaParallel - start_cudaParallel)) / CLOCKS_PER_SEC;

    // Copy result back to host
    cudaMemcpy(host_result, d_result, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    //check for error
    for(int i=0; i<SIZE; i++){
        if(serial_result[i]!= parallel_result[i]){
            std::cout<<"Error: results dont match at index: " << i << std::endl;
            return -1;
        }
    }

    for (int i = 0; i < SIZE; i++) {
        if (serial_result[i] != host_result[i]) {
            std::cerr << "Error: Results don't match at index " << i << std::endl;
            return -1;
        }
    }

    std::cout<<"The time for serial execution is "<< time_serial<< "seconds"<<std::endl;
    std::cout<<"The time for parallel openmp execution is "<< time_para<< "seconds"<<std::endl;
    std::cout<<"The time for parallel CUDA execution is "<< time_cudaParallel<< "seconds"<<std::endl;
    std::cout<<"serial- openmp Speed-up is "<< time_serial/time_para <<std::endl;
    std::cout<<"serial- CUDA Speed-up is "<< time_serial/time_cudaParallel <<std::endl;
    std::cout<<"openmp- CUDA Speed-up is "<< time_para/time_cudaParallel <<std::endl;


    delete[] a;
    delete[] b;
    delete[] serial_result;
    delete[] parallel_result;
    delete[] host_result;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}