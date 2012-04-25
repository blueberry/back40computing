#include <stdio.h>


#include <b40/reduction.cuh>


int main()
{
	typedef int T;

	T *d_in = NULL;
	T *d_out = NULL;

	b40::Reduce(d_in, d_out, 5);

	return 0;
}
