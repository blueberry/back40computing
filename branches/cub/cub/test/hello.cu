

#include <stdio.h>

#include <cub/io.cuh>

using namespace cub;

template <ReadModifier READ_MODIFIER, typename T>
__global__ void Kernel(T *d_in, T *d_out)
{
	T datum = Load<READ_MODIFIER>(d_in);
	*d_out = datum;
}


__global__ void Kernel2(double4 *d_in, double4 *d_out)
{
	double4 datum = *d_in;
	*d_out = datum;
}


/**
 * Main
 */
int main(int argc, const int**argv)
{
	double* d_double = NULL;
	Kernel<READ_CS><<<1,1>>>(d_double, d_double);

	double1* d_double1 = NULL;
	Kernel<READ_CS><<<1,1>>>(d_double1, d_double1);

	double2* d_double2 = NULL;
	Kernel<READ_CS><<<1,1>>>(d_double2, d_double2);

	double4* d_double4 = NULL;
	Kernel<READ_CS><<<1,1>>>(d_double4, d_double4);

	Kernel2<<<1,1>>>(d_double4, d_double4);

	cudaDeviceSynchronize();

	return 0;

}
