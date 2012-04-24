

#include <stdio.h>
#include <stdlib.h>


/******************************************************************************
 *
 *****************************************************************************/

#ifndef __CUDA_ARCH__
	#define __CUB_CUDA_ARCH__ 0						// Host path
#else
	#define __CUB_CUDA_ARCH__ __CUDA_ARCH__			// Device path
#endif


template <typename T>
__global__ void EmptyKernel(void) {}


/******************************************************************************
 * Kernel tuning policy and entrypoint
 *****************************************************************************/

/**
 * "Foo" kernel tuning policy
 */
template <
	int _THREADS,
	int _STRIPS_PER_THREAD,
	int _ELEMENTS_PER_STRIP>
struct KernelPolicy
{
	enum {
		// Reflected parameters
		THREADS					= _THREADS,
		STRIPS_PER_THREAD		= _STRIPS_PER_THREAD,
		ELEMENTS_PER_STRIP		= _ELEMENTS_PER_STRIP,

		// Derived
		TILE_SIZE 				= THREADS * STRIPS_PER_THREAD * ELEMENTS_PER_STRIP,
	};
};


/**
 * "Foo" kernel entrypoint
 */
template <typename KernelPolicy, typename T>
__global__ void FooKernel(T *d_data)
{
	d_data[0] = KernelPolicy::TILE_SIZE;
}


/******************************************************************************
 * Collection of tuned Foo specializations for different problem instances
 *****************************************************************************/

/**
 * Tuned policy specializations
 */
template <int TUNED_ARCH, typename T>
struct TunedPolicy {};

// 100
template <typename T>
struct TunedPolicy<100, T> : KernelPolicy<64, 1, 1> {};

// 200
template <typename T>
struct TunedPolicy<200, T> : KernelPolicy<128, 2, 8> {};

// 300
template <typename T>
struct TunedPolicy<300, T> : KernelPolicy<128, 4, 32> {};


/******************************************************************************
 *
 *****************************************************************************/

/**
 * Foo dispatch assistant
 */
template <typename T>
struct FooDispatch
{
	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	// Determine the appropriate tuning arch-id from the arch-id targeted
	// by the active compiler pass.
	enum {
		TUNE_ARCH =
			(__CUB_CUDA_ARCH__ >= 300) ?
				300 :
				(__CUB_CUDA_ARCH__ >= 200) ?
					200 :
					100,
	};

	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	// Type signature of Foo's kernel entrypoint
	typedef void (*KernelPtr)(T *);

	// Tuning policy specific to the arch-id of the active compiler
	// pass.  (The policy's type signature is "opaque" to the target
	// architecture.)
	struct OpaquePolicy : TunedPolicy<TUNE_ARCH, T> {};

	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	KernelPtr 		kernel_ptr;
	int 			tile_size;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	// Initializer
	template <typename KernelPolicy>
	void Init(KernelPtr kernel_ptr)
	{
		this->kernel_ptr = kernel_ptr;
		tile_size = KernelPolicy::TILE_SIZE;
	}

	// Constructor.  Initializes kernel pointer and reflective fields using
	// the supplied policy type.
	template <typename KernelPolicy>
	FooDispatch(KernelPolicy policy)
	{
		Init<KernelPolicy>(FooKernel<KernelPolicy>);
	}

	// Constructor.  Initializes kernel pointer and reflective fields using
	// an appropriate policy specialization for the given ptx version.
	FooDispatch(int ptx_version)
	{
		KernelPtr kernel_ptr = FooKernel<OpaquePolicy>;

		if (ptx_version >= 300) {
			Init<TunedPolicy<300, T> >(kernel_ptr);

		} else if (ptx_version >= 200) {
			Init<TunedPolicy<200, T> >(kernel_ptr);

		} else {
			Init<TunedPolicy<100, T> >(kernel_ptr);
		}
	}
};








/******************************************************************************
 * Host Program
 *****************************************************************************/


/**
 * Test selection and dispatch of autotuned policy
 */
template <typename T>
void TestAutotunedPolicy(T *d_data)
{
	cudaError_t error;

	// Get PTX version of compiled kernel assemblies (using EmptyKernel)
	cudaFuncAttributes empty_attrs;
	if (error = cudaFuncGetAttributes(&empty_attrs, EmptyKernel<void>))
	{
		printf("No valid PTX, error(%d: %s)\n",
			error,
			cudaGetErrorString(error));
		exit(1);
	}
	int ptx_version = empty_attrs.ptxVersion * 10;
	printf("Ptx version(%d)\n", ptx_version);

	// Dispatch kernel
	FooDispatch<T> foo_dispatch(ptx_version);
	foo_dispatch.kernel_ptr<<<1,1>>>(d_data);

	// Copy results back
	T h_data;
	if (error = cudaMemcpy(&h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost)) {
		printf("cudaMemcpy failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	printf("Autotuned Policy h_data(%d)\n", h_data);
	fflush(stdout);
}


/**
 * Test custom tuning policy
 */
template <typename T>
void TestCustomPolicy(T *d_data)
{
	KernelPolicy<256, 2, 1> custom_policy;

	cudaError_t error;

	// Dispatch kernel
	FooDispatch<T> foo_dispatch(custom_policy);
	foo_dispatch.kernel_ptr<<<1,1>>>(d_data);

	// Copy results back
	T h_data;
	if (error = cudaMemcpy(&h_data, d_data, sizeof(h_data), cudaMemcpyDeviceToHost)) {
		printf("cudaMemcpy failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	printf("Custom Policy h_data(%d)\n", h_data);
	fflush(stdout);
}


/**
 * Main
 */
int main(int argc, const char**argv)
{
	typedef int T;

	cudaError_t error;

	// Set device
	int current_device = 0;
	if (argc > 1) {
		current_device = atoi(argv[1]);
	}
	printf("Device(%d)\n", current_device);
	if (error = cudaSetDevice(current_device)) {
		printf("cudaSetDevice failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	// Allocate device data
	T *d_data;
	if (error = cudaMalloc((void**) &d_data, sizeof(T))) {
		printf("cudaMalloc failed (%d:%s)\n",  error, cudaGetErrorString(error));
		exit(1);
	}

	// Test autotuned policy
	TestAutotunedPolicy(d_data);

	// Test custom policy
	TestCustomPolicy(d_data);

	return 0;
}
