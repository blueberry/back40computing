/******************************************************************************
 *
 * Duane Merrill
 * 4/30/12
 *
 *
 *****************************************************************************/

#include <stdio.h>
#include <algorithm>

#include <thrust/device_func.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

#include "b40c_test_util.h"


/******************************************************************************
 * Misc. utilities
 *****************************************************************************/

typedef unsigned int uint;

/**
 * Mark's pointer wrapper
 */
namespace my_dev
{
	template <typename T>
	struct dev_mem
	{
		T *raw;

		dev_mem(T *raw) : raw(raw) {}

		T* raw_p()
		{
			return raw;
		}
	};
} // namespace my_dev


/**
 * 96-bit uint4 comparator
 */
bool Uint4Compare96(uint4 elem1, uint4 elem2)
{
	if (elem1.x != elem2.x) {
		return (elem1.x < elem2.x);

	} else if (elem1.y != elem2.y) {
		return (elem1.y < elem2.y);

	} else {
		return (elem1.z < elem2.z);
	}
}


/******************************************************************************
 * 96-bit sorting code
 *****************************************************************************/

namespace original
{

	/**
	 * Extract 32-bit word from uint4
	 */
	template<int keyIdx>
	struct ExtractBits: public thrust::unary_function<uint4, uint>
	{
		__host__ __device__ __forceinline__ uint operator()(uint4 key) const
		{
			if (keyIdx == 0)
				return key.x;
			else if (keyIdx == 1)
				return key.y;
			else
				return key.z;
		}
	};


	/**
	 * Update permutation
	 */
	template<
		int keyIdx,
		typename KeyPtr,
		typename PermutationPtr,
		typename ExtractedPtr>
	void update_permutation(
		KeyPtr& keys,
		PermutationPtr& permutation,
		ExtractedPtr& temp, int N)
	{
		// permute the keys with the current reordering
		thrust::gather(
			permutation,
			permutation + N,
			thrust::make_transform_iterator(
				keys,
				ExtractBits<keyIdx> ()),
			temp);

		// stable_sort the permuted keys and update the permutation
		thrust::stable_sort_by_key(
			temp,
			temp + N,
			permutation);
	}


	/**
	 * Apply permutation
	 */
	template<
		typename KeyPtr,
		typename PermutationPtr,
		typename OutputPtr>
	void apply_permutation(
		KeyPtr& keys,
		PermutationPtr& permutation,
		OutputPtr& out,
		int N)
	{
		// permute the keys into out vector
		thrust::gather(
			permutation,
			permutation + N,
			keys,
			out);
	}


	/**
	 * Sort the lower 96-bits of a uint4 structure
	 */
	void thrust_sort_96b(
		my_dev::dev_mem<uint4> srcKeys,
		my_dev::dev_mem<uint4> sortedKeys,
		my_dev::dev_mem<uint> temp_buffer,
		my_dev::dev_mem<uint> permutation_buffer,
		int N)
	{

		// wrap raw pointer with a device_func
		thrust::device_func<uint4> keys = thrust::device_pointer_cast(
				srcKeys.raw_p());
		thrust::device_func<uint4> outKeys = thrust::device_pointer_cast(
				sortedKeys.raw_p());
		thrust::device_func<uint> temp = thrust::device_pointer_cast(
				temp_buffer.raw_p());
		thrust::device_func<uint> permutation = thrust::device_pointer_cast(
				permutation_buffer.raw_p());

		// initialize permutation to [0, 1, 2, ... ,N-1]
		thrust::sequence(permutation, permutation + N);

		// sort z, y, x
		// careful: note 2, 1, 0 key word order, NOT 0, 1, 2.
		update_permutation<2> (keys, permutation, temp, N);
		update_permutation<1> (keys, permutation, temp, N);
		update_permutation<0> (keys, permutation, temp, N);

		// Note: keys have not been modified
		// Note: permutation now maps unsorted keys to sorted order

		thrust::gather(permutation, permutation + N, keys, outKeys);
	}

} // namespace original



/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_elements 	= 450 * 1000;	// 450K 96-bit keys
	int device_id 		= 0;

	// Get device id from command line
	if (argc > 1) {
		device_id = atoi(argv[1]);
	}
	cudaSetDevice(device_id);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device_id);
	printf("Using device %d: %s\n", device_id, deviceProp.name);

    // Allocate and initialize 96-bit keys on host
	printf("Allocating...\n"); fflush(stdout);
    uint4 *h_keys = new uint4[num_elements];
    for (int i(0); i < num_elements; ++i) {

    	b40c::util::RandomBits(h_keys[i].x);
    	b40c::util::RandomBits(h_keys[i].y);
    	b40c::util::RandomBits(h_keys[i].z);
    }

    // Compute answer (sorted keys) on host
	printf("Computing reference answer...\n"); fflush(stdout);
    uint4 *h_reference_keys = new uint4[num_elements];
    memcpy(h_reference_keys, h_keys, sizeof(uint4) * num_elements);
	std::stable_sort(h_reference_keys, h_reference_keys + num_elements, Uint4Compare96);

    // Allocate keys on device
	printf("Allocating problem to GPU...\n"); fflush(stdout);
    uint4 *d_keys;
    cudaMalloc((void**)&d_keys, sizeof(uint4) * num_elements);

	// Allocate sorted keys on device
    uint4 *d_sorted_keys;
	cudaMalloc((void**)&d_sorted_keys, sizeof(uint4) * num_elements);

    // Allocate output permutation vector on device
	uint *d_permutation;
	cudaMalloc((void**)&d_permutation, sizeof(uint) * num_elements);

	// Allocate temp buffers on device
	uint *d_temp0;
	uint *d_temp1;
	cudaMalloc((void**)&d_temp0, sizeof(uint) * num_elements);
	cudaMalloc((void**)&d_temp1, sizeof(uint) * num_elements);


	//
    // Thrust
	//

	// Copy problem to GPU
	printf("Thrust: copying problem to GPU...\n"); fflush(stdout);
	cudaMemcpy(d_keys, h_keys, sizeof(uint4) * num_elements, cudaMemcpyHostToDevice);

	// Thrust sort
	original::thrust_sort_96b(
		my_dev::dev_mem<uint4>(d_keys),
		my_dev::dev_mem<uint4>(d_sorted_keys),
		my_dev::dev_mem<uint>(d_temp1),
		my_dev::dev_mem<uint>(d_permutation),
		num_elements);

	// Copy out results and check answer
    uint4 *h_sorted_keys = new uint4[num_elements];
	cudaMemcpy(h_sorted_keys, d_sorted_keys, sizeof(uint4) * num_elements, cudaMemcpyDeviceToHost);
	bool correct = true;
	for (int i(0); i < num_elements; ++i) {

		if ((h_sorted_keys[i].z != h_reference_keys[i].z) ||
			(h_sorted_keys[i].y != h_reference_keys[i].y) ||
			(h_sorted_keys[i].x != h_reference_keys[i].x))
		{
			printf("Incorrect: [%d]: (%d,%d,%d) != (%d,%d,%d)\n",
				i,
				h_sorted_keys[i].z,
				h_sorted_keys[i].y,
				h_sorted_keys[i].x,
				h_reference_keys[i].z,
				h_reference_keys[i].y,
				h_reference_keys[i].x);

			correct = false;
			break;
		}
	}
	if (correct) {
		printf("Correct\n");
	}

    // Cleanup
    delete h_keys;
    delete h_sorted_keys;
    delete h_reference_keys;

    cudaFree(d_keys);
    cudaFree(d_sorted_keys);
    cudaFree(d_permutation);
    cudaFree(d_temp0);
    cudaFree(d_temp1);

    return 0;

}
