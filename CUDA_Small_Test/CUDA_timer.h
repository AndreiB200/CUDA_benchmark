#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include <chrono>


cudaEvent_t start_e, stop_e;
void createTimer()
{
	//cudaEvent_t start, stop;
	cudaEventCreate(&start_e);
	cudaEventCreate(&stop_e);
}

void startTimer()
{
	cudaEventRecord(start_e, 0);
}

float stopTimer()
{
	cudaEventRecord(stop_e, 0);
	cudaEventSynchronize(stop_e);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start_e, stop_e);

	return elapsedTime;
}

void cleanTimer()
{
	cudaEventDestroy(start_e);
	cudaEventDestroy(stop_e);
}


float timeCPU = 0.0f;
std::chrono::system_clock::time_point start_CPU;
std::chrono::system_clock::time_point end_CPU;

void startClock()
{
	start_CPU = std::chrono::system_clock::now();
}

float stopClock()
{
	end_CPU = std::chrono::system_clock::now();
	float durationCount = 0.0f;

	durationCount = std::chrono::duration<float, std::milli>(end_CPU - start_CPU).count();

	timeCPU = durationCount;

	return durationCount;
}