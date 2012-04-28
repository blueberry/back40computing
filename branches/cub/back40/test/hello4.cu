

#include <stdio.h>

#include <cub/type_utils.cuh>

using namespace cub;

void Foo1(int2 bar) {

}

void Foo2(int2 *bar) {

}

void Foo3(int2 &bar) {

}


int main()
{
	VectorT<int, 2> b;
	b.x = 1;
	b.y = 3;

	Foo1(b);
	Foo2(&b);
	Foo3(b);

	typedef int A[10][2];

	printf("sizeof basetype(%d), dims(%d), elements(%d)\n",
		sizeof(typename ArrayTraits<A>::Type),
		ArrayTraits<A>::DIMS,
		ArrayTraits<A>::ELEMENTS);

	typedef int* B;

	printf("sizeof basetype(%d), dims(%d), elements(%d)\n",
		sizeof(typename ArrayTraits<B>::Type),
		ArrayTraits<B>::DIMS,
		ArrayTraits<B>::ELEMENTS);

	return 0;
}
