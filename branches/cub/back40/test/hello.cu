

#include <stdio.h>

#include <cub/cub.cuh>


using namespace cub;

template <
	LoadModifier LOAD_MODIFIER,
	StoreModifier STORE_MODIFIER,
	typename T>
__global__ void Kernel(T *d_in, T *d_out)
{
	T datum = Load<LOAD_MODIFIER>(d_in + threadIdx.x);
	Store<STORE_MODIFIER>(d_out + threadIdx.x, datum);

	datum = Load<LOAD_MODIFIER>(d_in + threadIdx.x + 1);
	Store<STORE_MODIFIER>(d_out + threadIdx.x + 1, datum);

}

struct Foo { int a; double b; } *d_struct = NULL;

__global__ void Kernel2(Foo *d_in, Foo *d_out)
{
	Foo datum = *(d_in + threadIdx.x);
	*(d_out + threadIdx.x) = datum;

	datum = *(d_in + threadIdx.x + 1);
	*(d_out + threadIdx.x + 1) = datum;
}


/**
 * Main
 */
int main(int argc, const int**argv)
{
/*
	double* d_double = NULL;
	Kernel<LOAD_CS, STORE_WB><<<1,1>>>(d_double, d_double);

	double1* d_double1 = NULL;
	Kernel<LOAD_CS, STORE_WB><<<1,1>>>(d_double1, d_double1);

	double2* d_double2 = NULL;
	Kernel<LOAD_CS, STORE_WB><<<1,1>>>(d_double2, d_double2);

	double4* d_double4 = NULL;
	Kernel<LOAD_CG, STORE_CG><<<1,1>>>(d_double4, d_double4);

	Kernel2<<<1,1>>>(d_double4, d_double4);
*/

	Kernel<LOAD_CG, STORE_CG><<<1,1>>>(d_struct, d_struct);

	return 0;

}
