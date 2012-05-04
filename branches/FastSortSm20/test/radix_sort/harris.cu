/******************************************************************************
 *
 * Duane Merrill
 * 4/30/12
 *
 *
 *****************************************************************************/

#include <stdio.h>
#include <algorithm>

#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

// Sorting includes
#include <b40c/util/multi_buffer.cuh>
#include <b40c/radix_sort/enactor.cuh>

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
 * 96-bit sorting code (Original)
 *****************************************************************************/

namespace harris_original
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
		ExtractedPtr& temp,
		int N)
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
		thrust::stable_sort_by_key(temp, temp + N, permutation);
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
		thrust::gather(permutation, permutation + N, keys, out);
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

		// wrap raw pointer with a device_ptr
		thrust::device_ptr<uint4> keys = thrust::device_pointer_cast(
				srcKeys.raw_p());
		thrust::device_ptr<uint4> outKeys = thrust::device_pointer_cast(
				sortedKeys.raw_p());
		thrust::device_ptr<uint> temp = thrust::device_pointer_cast(
				temp_buffer.raw_p());
		thrust::device_ptr<uint> permutation = thrust::device_pointer_cast(
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

		apply_permutation(keys, permutation, outKeys, N);
	}


	/**
	 * Test interface
	 */
	void Test(
		uint4 *h_keys,
		uint4 *h_sorted_keys,
		uint4 *h_reference_keys,
		uint4 *d_keys,
		uint4 *d_sorted_keys,
		int N,
		int iterations)
	{
	    // Allocate permutation vector on device
		uint *d_permutation;
		cudaMalloc((void**)&d_permutation, sizeof(uint) * N);

	    // Allocate temporary buffer for 32-bit keys on device
		uint *d_temp;
		cudaMalloc((void**)&d_temp, sizeof(uint) * N);

		// Create timer and run for specified iterations
		b40c::GpuTimer gpu_timer;
		float elapsed = 0;
		for (int i = 0; i < iterations; ++i)
		{
			// Copy problem to GPU
			cudaMemcpy(d_keys, h_keys, sizeof(uint4) * N, cudaMemcpyHostToDevice);

			gpu_timer.Start();

			// Thrust sort
			thrust_sort_96b(
				my_dev::dev_mem<uint4>(d_keys),
				my_dev::dev_mem<uint4>(d_sorted_keys),
				my_dev::dev_mem<uint>(d_temp),
				my_dev::dev_mem<uint>(d_permutation),
				N);

			gpu_timer.Stop();
			elapsed += gpu_timer.ElapsedMillis();
		}
		float avg_elapsed = elapsed / float(iterations);
		printf("Total elapsed millis: %f, avg millis: %f, throughput: %.2f Mkeys/s\n",
			elapsed,
			avg_elapsed,
			float(N) / avg_elapsed / 1000.f);

		// Copy out results and check answer
		cudaMemcpy(h_sorted_keys, d_sorted_keys, sizeof(uint4) * N, cudaMemcpyDeviceToHost);
		bool correct = true;
		for (int i(0); i < N; ++i) {

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
	    cudaFree(d_permutation);
	    cudaFree(d_temp);
	}


} // namespace harris_original



/******************************************************************************
 * 96-bit sorting code (Back40)
 *****************************************************************************/

namespace harris_back40
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
	template<int keyIdx, typename KeyPtr>
	void update_permutation(
		KeyPtr& keys,
		int N,
		b40c::util::DoubleBuffer<uint, uint> &double_buffer,
		b40c::radix_sort::Enactor &sort_enactor)
	{
		// thrust ptr to permutation buffer
		thrust::device_ptr<uint> permutation = thrust::device_pointer_cast(
			double_buffer.d_values[double_buffer.selector]);

		// thrust ptr to temporary 32-bit keys
		thrust::device_ptr<uint> temp = thrust::device_pointer_cast(
			double_buffer.d_keys[double_buffer.selector]);

		// gather into temporary keys with the current reordering
		thrust::gather(
			permutation,
			permutation + N,
			thrust::make_transform_iterator(
				keys,
				ExtractBits<keyIdx> ()),
			temp);

		// Stable-sort the top 30 bits of the temp keys (and
		// associated permutation values)
		sort_enactor.Sort<30, 0>(double_buffer, N);
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
		thrust::gather(permutation, permutation + N, keys, out);
	}


	/**
	 * Sort the lower 96-bits of a uint4 structure
	 */
	void back40_sort_96b(
		my_dev::dev_mem<uint4> srcKeys,
		my_dev::dev_mem<uint4> sortedKeys,
		int N,
		b40c::util::DoubleBuffer<uint, uint> &double_buffer,
		b40c::radix_sort::Enactor &sort_enactor)
	{

		// thrust ptr to keys
		thrust::device_ptr<uint4> keys = thrust::device_pointer_cast(
				srcKeys.raw_p());

		// thrust ptr to permutation buffer
		thrust::device_ptr<uint> permutation = thrust::device_pointer_cast(
				double_buffer.d_values[double_buffer.selector]);

		// initialize values (permutation) to [0, 1, 2, ... ,N-1]
		thrust::sequence(permutation, permutation + N);

		// sort z, y, x
		// careful: note 2, 1, 0 key word order, NOT 0, 1, 2.
		update_permutation<2> (keys, N, double_buffer, sort_enactor);
		update_permutation<1> (keys, N, double_buffer, sort_enactor);
		update_permutation<0> (keys, N, double_buffer, sort_enactor);

		// Note: keys have not been modified

		// refresh thrust ptr to permutation buffer (may have changed inside ping-pong)
		permutation = thrust::device_pointer_cast(
				double_buffer.d_values[double_buffer.selector]);

		// Note: permutation now maps unsorted keys to sorted order

		thrust::device_ptr<uint4> outKeys = thrust::device_pointer_cast(
				sortedKeys.raw_p());

		apply_permutation(keys, permutation, outKeys, N);
	}


	/**
	 * Test interface
	 */
	void Test(
		uint4 *h_keys,
		uint4 *h_sorted_keys,
		uint4 *h_reference_keys,
		uint4 *d_keys,
		uint4 *d_sorted_keys,
		int N,
		int iterations)
	{
		// Allocate reusable ping-pong buffers on device.  The key buffers are opaque.
		b40c::util::DoubleBuffer<uint, uint> double_buffer;
		cudaMalloc((void**) &double_buffer.d_keys[0], sizeof(uint) * N);
		cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(uint) * N);

		// The current value buffer (double_buffer.d_keys[double_buffer.selector])
		// backs the desired permutation vector.
		cudaMalloc((void**) &double_buffer.d_values[0], sizeof(uint) * N);
		cudaMalloc((void**) &double_buffer.d_values[1], sizeof(uint) * N);

		// Create a reusable sorting enactor
		b40c::radix_sort::Enactor sort_enactor;

		// Create timer and run for specified iterations
		b40c::GpuTimer gpu_timer;
		float elapsed = 0;
		for (int i = 0; i < iterations; ++i)
		{
			// Copy problem to GPU
			cudaMemcpy(d_keys, h_keys, sizeof(uint4) * N, cudaMemcpyHostToDevice);

			gpu_timer.Start();

			// Back40 sort
			back40_sort_96b(
				my_dev::dev_mem<uint4>(d_keys),
				my_dev::dev_mem<uint4>(d_sorted_keys),
				N,
				double_buffer,
				sort_enactor);

			gpu_timer.Stop();
			elapsed += gpu_timer.ElapsedMillis();
		}
		float avg_elapsed = elapsed / float(iterations);
		printf("Total elapsed millis: %f, avg millis: %f, throughput: %.2f Mkeys/s\n",
			elapsed,
			avg_elapsed,
			float(N) / avg_elapsed / 1000.f);

		// Copy out results and check answer
	    h_sorted_keys = new uint4[N];
		cudaMemcpy(h_sorted_keys, d_sorted_keys, sizeof(uint4) * N, cudaMemcpyDeviceToHost);
		bool correct = true;
		for (int i(0); i < N; ++i) {

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
		if (double_buffer.d_keys[0]) cudaFree(double_buffer.d_keys[0]);
		if (double_buffer.d_keys[1]) cudaFree(double_buffer.d_keys[1]);
		if (double_buffer.d_values[0]) cudaFree(double_buffer.d_values[0]);
		if (double_buffer.d_values[1]) cudaFree(double_buffer.d_values[1]);

	}

} // namespace harris_back40




/**
 * Main
 */
int main(int argc, char** argv)
{
    int N 						= 450 * 1000;	// 450K 96-bit keys
	int device_id 				= 0;
	int iterations				= 100;

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
    uint4 *h_keys = new uint4[N];
    uint4 *h_sorted_keys = new uint4[N];
    for (int i(0); i < N; ++i) {

    	b40c::util::RandomBits(h_keys[i].x, 0, 30);
    	b40c::util::RandomBits(h_keys[i].y, 0, 30);
    	b40c::util::RandomBits(h_keys[i].z, 0, 30);
    }

    // Compute answer (sorted keys) on host
	printf("Computing reference answer...\n"); fflush(stdout);
    uint4 *h_reference_keys = new uint4[N];
    memcpy(h_reference_keys, h_keys, sizeof(uint4) * N);
	std::stable_sort(h_reference_keys, h_reference_keys + N, Uint4Compare96);

    // Allocate keys on device
	printf("Allocating problem to GPU...\n"); fflush(stdout);
    uint4 *d_keys;
    cudaMalloc((void**)&d_keys, sizeof(uint4) * N);

	// Allocate sorted keys on device
    uint4 *d_sorted_keys;
	cudaMalloc((void**)&d_sorted_keys, sizeof(uint4) * N);

	// Test original
	harris_original::Test(h_keys, h_sorted_keys, h_reference_keys, d_keys, d_sorted_keys, N, iterations);

	// Test back40
	harris_back40::Test(h_keys, h_sorted_keys, h_reference_keys, d_keys, d_sorted_keys, N, iterations);

    // Cleanup
    delete h_keys;
    delete h_sorted_keys;
    delete h_reference_keys;

    cudaFree(d_keys);
    cudaFree(d_sorted_keys);

    return 0;
}
