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
typedef unsigned long long ulonglong;

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

		void set_raw_p(T* new_raw)
		{
			raw = new_raw;
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
 * Original Thrust 96-bit sorting: sorts all 32 bits in uint4's key
 * structure's x, y, z fields.
 *****************************************************************************/


/**
 * Apply thrust_permutation
 */
template<
	typename KeyPtr,
	typename PermutationPtr,
	typename OutputPtr>
void apply_permutation(
	KeyPtr& thrust_in,
	PermutationPtr& thrust_permutation,
	OutputPtr& thrust_out,
	int N)
{
	// permute the keys into out vector
	thrust::gather(
		thrust_permutation,
		thrust_permutation + N,
		thrust_in,
		thrust_out);
}


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


namespace thrust_uint4_96bit
{
	/**
	 * Update thrust_permutation
	 */
	template<
		int keyIdx,
		typename KeyPtr,
		typename PermutationPtr,
		typename ExtractedPtr>
	void update_permutation(
		KeyPtr& thrust_src_keys,
		PermutationPtr& thrust_permutation,
		ExtractedPtr& thust_32bit_temp,
		int N)
	{
		// permute the keys with the current reordering
		thrust::gather(
			thrust_permutation,
			thrust_permutation + N,
			thrust::make_transform_iterator(
				thrust_src_keys,
				ExtractBits<keyIdx> ()),
			thust_32bit_temp);

		// stable_sort the permuted keys and update the thrust_permutation
		thrust::stable_sort_by_key(
			thust_32bit_temp,
			thust_32bit_temp + N,
			thrust_permutation);
	}



	/**
	 * Original Thrust 96-bit sorting: sorts all 32 bits in uint4's key
     * structure's x, y, z fields.
	 */
	void thrust_sort_96b(
		my_dev::dev_mem<uint4> &srcKeys,
		my_dev::dev_mem<uint4> &sortedKeys,
		my_dev::dev_mem<uint> &temp_buffer,
		my_dev::dev_mem<uint> &permutation_buffer,
		int N)
	{
		// thrust ptr to srcKeys
		thrust::device_ptr<uint4> thrust_src_keys = thrust::device_pointer_cast(
				srcKeys.raw_p());

		// thrust ptr to sortedKeys
		thrust::device_ptr<uint4> thrust_out_keys = thrust::device_pointer_cast(
				sortedKeys.raw_p());

		// thrust ptr to permutation buffer
		thrust::device_ptr<uint> thrust_permutation = thrust::device_pointer_cast(
				permutation_buffer.raw_p());

		// thrust ptr to temporary 32-bit keys
		thrust::device_ptr<uint> thust_32bit_temp = thrust::device_pointer_cast(
				temp_buffer.raw_p());

		// initialize thrust_permutation to [0, 1, 2, ... ,N-1]
		thrust::sequence(thrust_permutation, thrust_permutation + N);

		// sort z, y, x
		// careful: note 2, 1, 0 key word order, NOT 0, 1, 2.
		update_permutation<2> (thrust_src_keys, thrust_permutation, thust_32bit_temp, N);
		update_permutation<1> (thrust_src_keys, thrust_permutation, thust_32bit_temp, N);
		update_permutation<0> (thrust_src_keys, thrust_permutation, thust_32bit_temp, N);

		// Note: thrust_permutation now maps unsorted keys to sorted order
		apply_permutation(thrust_src_keys, thrust_permutation, thrust_out_keys, N);
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
		printf("Thrust 96-bit sorting: (128-bit uint4 keys)\n");
		fflush(stdout);

		// Allocate thrust_permutation vector on device
		uint *d_permutation;
		cudaMalloc((void**)&d_permutation, sizeof(uint) * N);

	    // Allocate temporary buffer for 32-bit keys on device
		uint *d_temp;
		cudaMalloc((void**)&d_temp, sizeof(uint) * N);

		// Wrap using Mark-pointers
		my_dev::dev_mem<uint4> srcKeys(d_keys);
		my_dev::dev_mem<uint4> sortedKeys(d_sorted_keys);
		my_dev::dev_mem<uint> temp_buffer(d_temp);
		my_dev::dev_mem<uint> permutation_buffer(d_permutation);

		// Create timer and run for specified iterations
		b40c::GpuTimer gpu_timer;
		float elapsed = 0;
		for (int i = 0; i < iterations; ++i)
		{
			// Copy problem onto GPU
			cudaMemcpy(d_keys, h_keys, sizeof(uint4) * N, cudaMemcpyHostToDevice);

			gpu_timer.Start();

			// Thrust sort
			thrust_sort_96b(
				srcKeys,
				sortedKeys,
				temp_buffer,
				permutation_buffer,
				N);

			gpu_timer.Stop();
			elapsed += gpu_timer.ElapsedMillis();
		}
		float avg_elapsed = elapsed / float(iterations);
		printf("Total elapsed millis: %f, avg millis: %f, throughput: %.2f Mkeys/s\n\n",
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
			printf("Correct\n\n");
		}

	    // Cleanup
	    cudaFree(d_permutation);
	    cudaFree(d_temp);
	}


} // namespace thrust_uint4_96bit



/******************************************************************************
 * Back40 90-bit sorting: sorts the lower 30 bits in uint4's key
 * structure's x, y, z fields.
 *****************************************************************************/

namespace back40_uint4_90bit
{
	/**
	 * Update thrust_permutation
	 */
	template<int keyIdx, typename KeyPtr>
	void update_permutation(
		KeyPtr& thrust_src_keys,
		int N,
		b40c::util::DoubleBuffer<uint, uint> &double_buffer,
		b40c::radix_sort::Enactor &sort_enactor)
	{
		// thrust ptr to thrust_permutation buffer
		thrust::device_ptr<uint> thrust_permutation = thrust::device_pointer_cast(
			double_buffer.d_values[double_buffer.selector]);

		// thrust ptr to temporary 32-bit keys
		thrust::device_ptr<uint> thust_32bit_temp = thrust::device_pointer_cast(
			double_buffer.d_keys[double_buffer.selector]);

		// gather into temporary keys with the current reordering
		thrust::gather(
			thrust_permutation,
			thrust_permutation + N,
			thrust::make_transform_iterator(
				thrust_src_keys,
				ExtractBits<keyIdx> ()),
			thust_32bit_temp);

		// Stable-sort the top 30 bits of the temp keys (and
		// associated thrust_permutation values)
		sort_enactor.Sort<30, 0>(double_buffer, N);
	}


	/**
	 * Back40 90-bit sorting: sorts the lower 30 bits in uint4's key
     * structure's x, y, z fields.
	 */
	void back40_uint4_sort_90b(
		my_dev::dev_mem<uint4> &srcKeys,
		my_dev::dev_mem<uint4> &sortedKeys,
		int N,
		b40c::util::DoubleBuffer<uint, uint> &double_buffer,
		b40c::radix_sort::Enactor &sort_enactor)
	{
		// thrust ptr to srcKeys
		thrust::device_ptr<uint4> thrust_src_keys = thrust::device_pointer_cast(
				srcKeys.raw_p());

		// thrust ptr to sortedKeys
		thrust::device_ptr<uint4> thrust_out_keys = thrust::device_pointer_cast(
				sortedKeys.raw_p());

		// thrust ptr to permutation buffer
		thrust::device_ptr<uint> thrust_permutation = thrust::device_pointer_cast(
				double_buffer.d_values[double_buffer.selector]);

		// initialize values (thrust_permutation) to [0, 1, 2, ... ,N-1]
		thrust::sequence(thrust_permutation, thrust_permutation + N);

		// sort z, y, x
		// careful: note 2, 1, 0 key word order, NOT 0, 1, 2.
		update_permutation<2> (thrust_src_keys, N, double_buffer, sort_enactor);
		update_permutation<1> (thrust_src_keys, N, double_buffer, sort_enactor);
		update_permutation<0> (thrust_src_keys, N, double_buffer, sort_enactor);

		// refresh thrust ptr to permutation buffer (may have changed inside ping-pong)
		thrust_permutation = thrust::device_pointer_cast(
				double_buffer.d_values[double_buffer.selector]);

		// Note: thrust_permutation now maps unsorted keys to sorted order
		apply_permutation(thrust_src_keys, thrust_permutation, thrust_out_keys, N);
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
		printf("Back40 90-bit sorting: (128-bit uint4 keys)\n");
		fflush(stdout);

		// Allocate reusable ping-pong buffers on device.
		b40c::util::DoubleBuffer<uint, uint> double_buffer;

		// The key buffers are opaque.
		cudaMalloc((void**) &double_buffer.d_keys[0], sizeof(uint) * N);
		cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(uint) * N);

		// The current value buffer (double_buffer.d_values[double_buffer.selector])
		// backs the desired permutation array.
		cudaMalloc((void**) &double_buffer.d_values[0], sizeof(uint) * N);
		cudaMalloc((void**) &double_buffer.d_values[1], sizeof(uint) * N);

		// Wrap using Mark-pointers
		my_dev::dev_mem<uint4> srcKeys(d_keys);
		my_dev::dev_mem<uint4> sortedKeys(d_sorted_keys);

		// Create a reusable sorting enactor
		b40c::radix_sort::Enactor sort_enactor;

		// Create timer and run for specified iterations
		b40c::GpuTimer gpu_timer;
		float elapsed = 0;
		for (int i = 0; i < iterations; ++i)
		{
			// Copy problem onto GPU
			cudaMemcpy(d_keys, h_keys, sizeof(uint4) * N, cudaMemcpyHostToDevice);

			gpu_timer.Start();

			// Back40 sort
			back40_uint4_sort_90b(
				srcKeys,
				sortedKeys,
				N,
				double_buffer,
				sort_enactor);

			gpu_timer.Stop();
			elapsed += gpu_timer.ElapsedMillis();
		}
		float avg_elapsed = elapsed / float(iterations);
		printf("Total elapsed millis: %f, avg millis: %f, throughput: %.2f Mkeys/s\n\n",
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
			printf("Correct\n\n");
		}

		// Cleanup
		if (double_buffer.d_keys[0]) cudaFree(double_buffer.d_keys[0]);
		if (double_buffer.d_keys[1]) cudaFree(double_buffer.d_keys[1]);
		if (double_buffer.d_values[0]) cudaFree(double_buffer.d_values[0]);
		if (double_buffer.d_values[1]) cudaFree(double_buffer.d_values[1]);

	}

} // namespace back40_uint4_90bit



/******************************************************************************
 * Back40 60-bit sorting: sorts the lower 60 bits in uint4's key
 * structure's x, y, z fields.
 *****************************************************************************/

namespace back40_uint4_60bit
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
	 * Update thrust_permutation
	 */
	template<int keyIdx, typename KeyPtr>
	void update_permutation(
		KeyPtr& thrust_src_keys,
		int N,
		b40c::util::DoubleBuffer<uint, uint> &double_buffer,
		b40c::radix_sort::Enactor &sort_enactor)
	{
		// thrust ptr to thrust_permutation buffer
		thrust::device_ptr<uint> thrust_permutation = thrust::device_pointer_cast(
			double_buffer.d_values[double_buffer.selector]);

		// thrust ptr to temporary 32-bit keys
		thrust::device_ptr<uint> thust_32bit_temp = thrust::device_pointer_cast(
			double_buffer.d_keys[double_buffer.selector]);

		// gather into temporary keys with the current reordering
		thrust::gather(
			thrust_permutation,
			thrust_permutation + N,
			thrust::make_transform_iterator(
				thrust_src_keys,
				ExtractBits<keyIdx> ()),
			thust_32bit_temp);

		// Stable-sort the top 20 bits of the temp keys (and
		// associated thrust_permutation values)
		sort_enactor.Sort<20, 0>(double_buffer, N);
	}


	/**
	 * Back40 60-bit sorting: sorts the lower 60 bits in uint4's key
	 * structure's x, y, z fields.
	 */
	void back40_uint4_sort_60b(
		my_dev::dev_mem<uint4> &srcKeys,
		my_dev::dev_mem<uint4> &sortedKeys,
		int N,
		b40c::util::DoubleBuffer<uint, uint> &double_buffer,
		b40c::radix_sort::Enactor &sort_enactor)
	{
		// thrust ptr to srcKeys
		thrust::device_ptr<uint4> thrust_src_keys = thrust::device_pointer_cast(
				srcKeys.raw_p());

		// thrust ptr to permutation buffer
		thrust::device_ptr<uint> thrust_permutation = thrust::device_pointer_cast(
				double_buffer.d_values[double_buffer.selector]);

		// thrust ptr to sortedKeys
		thrust::device_ptr<uint4> thrust_out_keys = thrust::device_pointer_cast(
				sortedKeys.raw_p());

		// initialize values (thrust_permutation) to [0, 1, 2, ... ,N-1]
		thrust::sequence(thrust_permutation, thrust_permutation + N);

		// sort z, y, x
		// careful: note 2, 1, 0 key word order, NOT 0, 1, 2.
		update_permutation<2> (thrust_src_keys, N, double_buffer, sort_enactor);
		update_permutation<1> (thrust_src_keys, N, double_buffer, sort_enactor);
		update_permutation<0> (thrust_src_keys, N, double_buffer, sort_enactor);

		// refresh thrust ptr permutation buffer (may have changed inside ping-pong)
		thrust_permutation = thrust::device_pointer_cast(
				double_buffer.d_values[double_buffer.selector]);

		// Note: thrust_permutation now maps unsorted keys to sorted order
		apply_permutation(thrust_src_keys, thrust_permutation, thrust_out_keys, N);
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
		printf("Back40 60-bit sorting: (128-bit uint4 keys)\n");
		fflush(stdout);

		// Allocate reusable ping-pong buffers on device.
		b40c::util::DoubleBuffer<uint, uint> double_buffer;

		// The key buffers are opaque.
		cudaMalloc((void**) &double_buffer.d_keys[0], sizeof(uint) * N);
		cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(uint) * N);

		// The current value buffer (double_buffer.d_values[double_buffer.selector])
		// backs the desired permutation array.
		cudaMalloc((void**) &double_buffer.d_values[0], sizeof(uint) * N);
		cudaMalloc((void**) &double_buffer.d_values[1], sizeof(uint) * N);

		// Wrap using Mark-pointers
		my_dev::dev_mem<uint4> srcKeys(d_keys);
		my_dev::dev_mem<uint4> sortedKeys(d_sorted_keys);

		// Create a reusable sorting enactor
		b40c::radix_sort::Enactor sort_enactor;

		// Create timer and run for specified iterations
		b40c::GpuTimer gpu_timer;
		float elapsed = 0;
		for (int i = 0; i < iterations; ++i)
		{
			// Copy problem onto GPU
			cudaMemcpy(d_keys, h_keys, sizeof(uint4) * N, cudaMemcpyHostToDevice);

			gpu_timer.Start();

			// Back40 sort
			back40_uint4_sort_60b(
				srcKeys,
				sortedKeys,
				N,
				double_buffer,
				sort_enactor);

			gpu_timer.Stop();
			elapsed += gpu_timer.ElapsedMillis();
		}
		float avg_elapsed = elapsed / float(iterations);
		printf("Total elapsed millis: %f, avg millis: %f, throughput: %.2f Mkeys/s\n\n",
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
			printf("Correct\n\n");
		}

		// Cleanup
		if (double_buffer.d_keys[0]) cudaFree(double_buffer.d_keys[0]);
		if (double_buffer.d_keys[1]) cudaFree(double_buffer.d_keys[1]);
		if (double_buffer.d_values[0]) cudaFree(double_buffer.d_values[0]);
		if (double_buffer.d_values[1]) cudaFree(double_buffer.d_values[1]);
	}

} // namespace back40_uint4_60bit



/******************************************************************************
 * Back40 60-bit sorting: sorts the lower 60 bits in a 64-bit ulonglong
 *****************************************************************************/

namespace back40_ulonglong_60bit
{
	/**
	 * Back40 60-bit sorting: sorts the lower 60 bits a 64-bit ulonglong
	 */
	void back40_ulonglong_sort_60bit(
		my_dev::dev_mem<ulonglong> &sortedKeys,
		int N,
		b40c::util::DoubleBuffer<ulonglong, uint> &double_buffer,
		b40c::radix_sort::Enactor &sort_enactor)
	{
		// thrust ptr to thrust_permutation buffer
		thrust::device_ptr<uint> thrust_permutation = thrust::device_pointer_cast(
			double_buffer.d_values[double_buffer.selector]);

		// initialize values (thrust_permutation) to [0, 1, 2, ... ,N-1]
		thrust::sequence(thrust_permutation, thrust_permutation + N);

		// Stable-sort the top 60 bits of the keys (and
		// associated thrust_permutation values)
		sort_enactor.Sort<60, 0>(double_buffer, N);

		// Note: double_buffer.d_values[double_buffer.selector] now maps unsorted keys to sorted order

		// refresh underlying raw pointer to sorted keys
		sortedKeys.set_raw_p(double_buffer.d_keys[double_buffer.selector]);
	}


	/**
	 * Test interface
	 */
	void Test(
		ulonglong *h_keys,
		ulonglong *h_sorted_keys,
		ulonglong *h_reference_keys,
		ulonglong *d_keys,
		int N,
		int iterations)
	{
		printf("Back40 60-bit sorting (64-bit ulonglong keys):\n");
		fflush(stdout);

		// Sorted keys pointer: we will set it inside back40_ulonglong_sort_60bit
		my_dev::dev_mem<ulonglong> d_sorted_keys(NULL);

		// Allocate reusable ping-pong buffers on device.
		b40c::util::DoubleBuffer<ulonglong, uint> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector])
		// backs the keys.
		double_buffer.d_keys[0] = d_keys;
		cudaMalloc((void**) &double_buffer.d_keys[1], sizeof(ulonglong) * N);

		// The current value buffer (double_buffer.d_values[double_buffer.selector])
		// backs the desired permutation array.
		cudaMalloc((void**) &double_buffer.d_values[0], sizeof(uint) * N);
		cudaMalloc((void**) &double_buffer.d_values[1], sizeof(uint) * N);

		// Create a reusable sorting enactor
		b40c::radix_sort::Enactor sort_enactor;

		// Create timer and run for specified iterations
		b40c::GpuTimer gpu_timer;
		float elapsed = 0;
		for (int i = 0; i < iterations; ++i)
		{
			// Copy problem onto GPU
			cudaMemcpy(d_keys, h_keys, sizeof(ulonglong) * N, cudaMemcpyHostToDevice);

			gpu_timer.Start();

			// Back40 sort
			back40_ulonglong_sort_60bit(
				d_sorted_keys,
				N,
				double_buffer,
				sort_enactor);

			gpu_timer.Stop();
			elapsed += gpu_timer.ElapsedMillis();
		}
		float avg_elapsed = elapsed / float(iterations);
		printf("Total elapsed millis: %f, avg millis: %f, throughput: %.2f Mkeys/s\n\n",
			elapsed,
			avg_elapsed,
			float(N) / avg_elapsed / 1000.f);

		// Copy out results and check answer
		h_sorted_keys = new ulonglong[N];
		cudaMemcpy(
			h_sorted_keys,
			d_sorted_keys.raw_p(),
			sizeof(ulonglong) * N, cudaMemcpyDeviceToHost);

		bool correct = true;
		for (int i(0); i < N; ++i) {

			if (h_sorted_keys[i] != h_reference_keys[i])
			{
				printf("Incorrect: [%d]: %d != %d\n",
					i,
					h_sorted_keys[i],
					h_reference_keys[i]);

				correct = false;
				break;
			}
		}
		if (correct) {
			printf("Correct\n\n");
		}

		// Cleanup
		if (double_buffer.d_keys[0]) cudaFree(double_buffer.d_keys[0]);
		if (double_buffer.d_keys[1]) cudaFree(double_buffer.d_keys[1]);
		if (double_buffer.d_values[0]) cudaFree(double_buffer.d_values[0]);
		if (double_buffer.d_values[1]) cudaFree(double_buffer.d_values[1]);

	}

} // namespace back40_ulonglong_60bit



/**
 * Main
 */
int main(int argc, char** argv)
{
    int N 						= 450 * 1000;	// 450K 96-bit keys
	int device_id 				= 0;
	int iterations				= 1000;

	// Get device id from command line
	if (argc > 1) {
		device_id = atoi(argv[1]);
	}
	cudaSetDevice(device_id);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device_id);
	printf("Evaluating %d iterations using device %d: %s\n\n",
		iterations,
		device_id,
		deviceProp.name);

	//
	// 128-bit uint4 key structure
	//
	{
		// Allocate and initialize 90-bit uint4 keys on host
		printf("Allocating 90-bit uint4 keys...\n"); fflush(stdout);
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
		printf("Allocating problem on GPU...\n"); fflush(stdout);
		uint4 *d_keys;
		cudaMalloc((void**)&d_keys, sizeof(uint4) * N);

		// Allocate sorted keys on device
		uint4 *d_sorted_keys;
		cudaMalloc((void**)&d_sorted_keys, sizeof(uint4) * N);

		// Test original (lower bits <= 30 bits in uint4's x,y,z)
		thrust_uint4_96bit::Test(
			h_keys,
			h_sorted_keys,
			h_reference_keys,
			d_keys,
			d_sorted_keys,
			N,
			iterations);

		// Test back40 (lower bits <= 30 bits in uint4's x,y,z)
		back40_uint4_90bit::Test(
			h_keys,
			h_sorted_keys,
			h_reference_keys,
			d_keys,
			d_sorted_keys,
			N,
			iterations);

		// Reduce active bits to 20 lower bits
		printf("Reducing to 20 active bits per channel...\n"); fflush(stdout);
		for (int i(0); i < N; ++i) {
			h_keys[i].x &= ((1 << 20) - 1);
			h_keys[i].y &= ((1 << 20) - 1);
			h_keys[i].z &= ((1 << 20) - 1);
		}

		// Compute answer (sorted keys) on host
		printf("Computing reference answer...\n"); fflush(stdout);
		memcpy(h_reference_keys, h_keys, sizeof(uint4) * N);
		std::stable_sort(h_reference_keys, h_reference_keys + N, Uint4Compare96);

		// Test back40 (lower bits <= 20 bits in uint4's x,y,z)
		back40_uint4_60bit::Test(
			h_keys,
			h_sorted_keys,
			h_reference_keys,
			d_keys,
			d_sorted_keys,
			N,
			iterations);

		// Cleanup
		delete h_keys;
		delete h_sorted_keys;
		delete h_reference_keys;
		cudaFree(d_keys);
		cudaFree(d_sorted_keys);
	}


	//
	// 64-bit ulonglong key structure
	//
	{
		// Allocate and initialize 60-bit ulong long keys on host
		printf("Allocating 60-bit ulonglong keys...\n"); fflush(stdout);
		ulonglong *h_keys = new ulonglong[N];
		ulonglong *h_sorted_keys = new ulonglong[N];
		for (int i(0); i < N; ++i) {
			b40c::util::RandomBits(h_keys[i], 0, 60);
		}

		// Compute answer (sorted keys) on host
		printf("Computing reference answer...\n"); fflush(stdout);
		ulonglong *h_reference_keys = new ulonglong[N];
		memcpy(h_reference_keys, h_keys, sizeof(ulonglong) * N);
		std::stable_sort(h_reference_keys, h_reference_keys + N);

		// Allocate keys on device
		printf("Allocating problem on GPU...\n"); fflush(stdout);
		ulonglong *d_keys;
		cudaMalloc((void**)&d_keys, sizeof(ulonglong) * N);

		// Test back40 (lower bits <= 60)
		back40_ulonglong_60bit::Test(
			h_keys,
			h_sorted_keys,
			h_reference_keys,
			d_keys,
			N,
			iterations);

		// Cleanup
		delete h_keys;
		delete h_sorted_keys;
		delete h_reference_keys;
		cudaFree(d_keys);

	}


    return 0;
}
