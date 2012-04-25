#include <stdio.h>


#include <b40/reduction.cuh>
#include <cub/tile/io.cuh>
#include <cub/core/operators.cuh>


int main()
{
	typedef int T;

	T *d_in = NULL;
	T *d_out = NULL;

	b40::Reduce(d_in, d_out, 5);

	b40::reduction::Policy<
		b40::reduction::KernelPolicy<32, 1, 1, cub::READ_NONE, cub::WRITE_NONE, false>,
		b40::reduction::KernelPolicy<32, 1, 1, cub::READ_NONE, cub::WRITE_NONE, false>,
		true,
		true> policy;

	cub::Sum<T> reduction_op;

	b40::Reduce(policy, d_in, d_out, 5, reduction_op);

	return 0;
}
