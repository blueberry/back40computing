#include <stdio.h>


#include <back40/back40.cuh>


int main()
{
	typedef int T;

	T *d_in = NULL;
	T *d_out = NULL;
	T *h_out = NULL;
	T *h_seed = NULL;

	back40::Reduce(d_in, d_out, h_out, h_seed, 5);
/*
	back40::reduce::Policy<
		back40::reduce::KernelPolicy<32, 1, 1, cub::READ_NONE, cub::WRITE_NONE, false>,
		back40::reduce::KernelPolicy<32, 1, 1, cub::READ_NONE, cub::WRITE_NONE, false>,
		true,
		true> policy;

	cub::Sum<T> reduction_op;

	back40::Reduce(d_in, d_out, h_out, h_seed, 5, reduction_op, policy);
*/
	return 0;
}
