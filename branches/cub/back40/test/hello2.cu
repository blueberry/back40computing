#include <stdio.h>


#include <cub/thread/reduce.cuh>

using namespace cub;


__forceinline__ int Shift(const int magnitude, int val)
{
	if (magnitude > 0) {
		return val << magnitude;
	} else {
		return val >> magnitude;
	}
}


int main()
{
	int a[4] 		= {1, 2, 3, 4};
	int b[2][2] 	= {{1, 2}, {3, 4}};
	int *c 			= a + 1;
	int (*d)[2] 	= b + 1;

	printf("%d\n", Reduce(a));		// 10
	printf("%d\n", Reduce(b));		// 10
	printf("%d\n", Reduce(c));		// 2
	printf("%d\n", Reduce<2>(c));	// 5
	printf("%d\n", Reduce(d));		// 7

	printf("%d\n", Shift(-2, 8));

	return 0;
}
