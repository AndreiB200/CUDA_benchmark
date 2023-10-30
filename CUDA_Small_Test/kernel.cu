
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "My_CUDA.cuh"

int main()
{
    cudaSetDevice(0);
    runCUDA();
    return 0;
}
