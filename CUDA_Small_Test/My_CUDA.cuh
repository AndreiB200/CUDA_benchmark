#include <cuda_runtime.h>

#include <iostream>
#include <time.h>
#include <chrono>
#include <vector>
#include <fstream>
#include "CUDA_timer.h"


#define DtoH cudaMemcpyDeviceToHost
#define HtoD cudaMemcpyHostToDevice
#define MEMCOPY_ITERATIONS 100


struct GPUTimers
{
    float totalGPU, kernel, transferHtoD, transferDtoH;
};

struct GlobalTimers
{
    float cpuTime;
    GPUTimers times;
};


__device__ float normal3dGPU(float a, float b, float c)
{
    float square = a * a + b * b + c * c;
    return sqrt(square);
}


__global__ void addKernel(int* c, int* a, int* b)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void invSqrt(float* d, float* a, float* b, float* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    d[i] = 1/normal3dGPU(a[i], b[i] ,c[i]);
}

GPUTimers setCudaComputing(int *a, int* b, int* c, int size)
{
    GPUTimers overallTime;
    int *device_a = 0;
    int *device_b = 0;
    int *device_c = 0;

    cudaMalloc((void**)&device_a, size * sizeof(int));
    cudaMalloc((void**)&device_b, size * sizeof(int));
    cudaMalloc((void**)&device_c, size * sizeof(int));

    startTimer();
    //Copy Host to Device
    cudaMemcpy(device_a, a, size * sizeof(int), HtoD);
    cudaMemcpy(device_b, b, size * sizeof(int), HtoD);

    float elapsedTime_mem_HostDevice = stopTimer();
    overallTime.transferHtoD = elapsedTime_mem_HostDevice;

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    
    startTimer();
    //Execute Kernel
    addKernel <<< blocksPerGrid, threadsPerBlock >>> (device_c, device_a, device_b);
    float elapsedTime_kernel = stopTimer();
    overallTime.kernel = elapsedTime_kernel;


    startTimer();
    //Copy Device to Host
    cudaMemcpy(c, device_c, size * sizeof(int), DtoH);
    float elapsedTime_DeviceHost = stopTimer();
    overallTime.transferDtoH = elapsedTime_DeviceHost;
    
    overallTime.totalGPU = elapsedTime_DeviceHost + elapsedTime_mem_HostDevice + elapsedTime_kernel;

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return overallTime;
}

GPUTimers setCudaHardComputing(float* a, float* b, float *c, float* d, int size)
{
    GPUTimers overallTime;
    float* device_a = 0;
    float* device_b = 0;
    float* device_c = 0;
    float* device_d = 0;
    cudaMalloc((void**)&device_a, size * sizeof(float));
    cudaMalloc((void**)&device_b, size * sizeof(float));
    cudaMalloc((void**)&device_c, size * sizeof(float));
    cudaMalloc((void**)&device_d, size * sizeof(float));

    startTimer();
    //Copy Host to Device
    cudaMemcpy(device_a, a, size * sizeof(float), HtoD);
    cudaMemcpy(device_b, b, size * sizeof(float), HtoD);
    cudaMemcpy(device_c, c, size * sizeof(float), HtoD);

    float elapsedTime_mem_HostDevice = stopTimer();
    overallTime.transferHtoD = elapsedTime_mem_HostDevice;

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    startTimer();
    //Execute Kernel
    invSqrt <<< blocksPerGrid, threadsPerBlock >>>(device_d, device_a, device_b, device_c);
    float elapsedTime_kernel = stopTimer();
    overallTime.kernel = elapsedTime_kernel;


    startTimer();
    //Copy Device to Host
    cudaMemcpy(d, device_d, size * sizeof(float), DtoH);
    float elapsedTime_DeviceHost = stopTimer();
    overallTime.transferDtoH = elapsedTime_DeviceHost;

    overallTime.totalGPU = elapsedTime_DeviceHost + elapsedTime_mem_HostDevice + elapsedTime_kernel;

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFree(device_d);

    return overallTime;
}

void cudaMemoryBandwidth(int* a, int size)
{
    float elapsedTotal = 0.0f;
    double bandwidthInGBs = 0.0;

    int* device_a = 0;
    cudaMalloc((void**)&device_a, size * sizeof(int));

    startTimer();
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        cudaMemcpy(device_a, a, size * sizeof(int), HtoD);

        std::cout << ".";
    }
    std::cout << std::endl;

    float elapsedTime = stopTimer();
    
    double time_s = elapsedTime / 1e3;
    bandwidthInGBs = (size * 2 * sizeof(int) * MEMCOPY_ITERATIONS) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    std::cout << bandwidthInGBs << " GB per sec" << std::endl;
   
    cudaFree(device_a);
}

void setCPU(int* a, int* b, int* d, int size)
{
    for (unsigned int i = 0; i < size; i++)
    {
        d[i] = a[i] + b[i];
    }
}

void setCPUhard(float* a, float* b, float* c, float* d, int size)
{
    for (unsigned int i = 0; i < size; i++)
    {
        d[i] = 1 / sqrt(a[i] * a[i] + b[i] * b[i] + c[i] * c[i]);
    }
}

void testCase(int* c, int size)
{
    for(int i = 0; i < size; i++)
        if (c[i] != 2)
        {
            std::cout << "ERROR ! at " << c[i] << std::endl;
            return;
        }
    std::cout << "TEST CORRECT" << std::endl;
}

void testCasehard(float* c, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (c[i] != 1 / sqrt(12))
        {
            std::cout << "ERROR ! at " << c[i] << std::endl;
            return;
        }
    }
    std::cout << "TEST CORRECT hard compute" << std::endl;
}

void runCUDA()
{
    createTimer();

    const int sizeTest = INT_MAX/8;
    int* x = (int*)malloc(sizeTest * sizeof(int));

    //cudaMemoryBandwidth(e, sizeTest);
    free(x);
    std::ofstream myfile;
    myfile.open("Data output.csv");

    myfile << "Simple Operations\n";
    myfile << "CPU,"; myfile << "GPU total,"; myfile << "GPU kernel,"; myfile << "GPU host to dev,"; myfile << "GPU dev to host,";
    myfile << "\n";

    int size = 1024;
    
    while (size < INT_MAX/4)
    {
        int* a = (int*)malloc(size * sizeof(int));
        int* b = (int*)malloc(size * sizeof(int));
        int* c = (int*)malloc(size * sizeof(int));//GPU addition
        int* d = (int*)malloc(size * sizeof(int));//CPU addition

        for (unsigned int i = 0; i < size; i++)
        {
            a[i] = 1;
            b[i] = 1;
        }

        startClock();
        setCPU(a, b, d, size);
        float cpuTime = stopClock();
        free(d);

        GlobalTimers tim;
        tim.cpuTime = cpuTime;
        tim.times = setCudaComputing(a, b, c, size);
        testCase(c, size);
        free(a); free(b); free(c);

        myfile << tim.cpuTime << ",";
        myfile << tim.times.totalGPU << ",";
        myfile << tim.times.kernel << ",";
        myfile << tim.times.transferHtoD << ",";
        myfile << tim.times.transferDtoH << ",";
        myfile << "\n";
        size = size * 2;
    }
    std::cout << "Press Enter for next benchmark..."; std::cin.get();

    myfile << "\n";
    myfile << "Complicated Operations\n";
    myfile << "CPU,"; myfile << "GPU total,"; myfile << "GPU kernel,"; myfile << "GPU host to dev,"; myfile << "GPU dev to host,";
    myfile << "\n";

    size = 1024;
    while (size < INT_MAX / 8)
    {
        float* a = (float*)malloc(size * sizeof(float));
        float* b = (float*)malloc(size * sizeof(float));
        float* c = (float*)malloc(size * sizeof(float));
        float* d = (float*)malloc(size * sizeof(float));//GPU addition
        float* e = (float*)malloc(size * sizeof(float));//CPU addition

        for (unsigned int i = 0; i < size; i++)
        {
            a[i] = 2.0f;
            b[i] = 2.0f;
            c[i] = 2.0f;
        }

        startClock();
        setCPUhard(a, b, c, e, size);
        float cpuTime = stopClock();
        free(e);

        GlobalTimers tim;
        tim.cpuTime = cpuTime;
        tim.times = setCudaHardComputing(a, b, c, d, size);
        testCasehard(d, size);
        free(a); free(b); free(c); free(d);

        myfile << tim.cpuTime << ",";
        myfile << tim.times.totalGPU << ",";
        myfile << tim.times.kernel << ",";
        myfile << tim.times.transferHtoD << ",";
        myfile << tim.times.transferDtoH << ",";
        myfile << "\n";
        size = size * 2;
    }


    myfile.close();
}
