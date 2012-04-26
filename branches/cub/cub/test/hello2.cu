

#include <stdio.h>
#include <stdlib.h>

#include <cub/cuda_properties.cuh>
#include <cub/kernel_properties.cuh>

/******************************************************************************
 * Kernel tuning policy and entrypoint
 *****************************************************************************/

/**
 * FooKernel tuning policy
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
 * Foo tuning policy
 */
template <
	typename UpsweepKernelPolicy,
	typename SpineKernelPolicy,
	typename DownsweepKernelPolicy,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE>
struct Policy
{
	typedef UpsweepKernelPolicy 		Upsweep;
	typedef SpineKernelPolicy 			Spine;
	typedef DownsweepKernelPolicy 		Downsweep;

	// Host-specific tuning details
	enum {
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
	};
};





/**
 * "Foo" kernel entrypoint
 */
template <typename KernelPolicy, typename T, typename SizeT>
__global__ void FooKernel(T *d_data, SizeT num_elements)
{
	d_data[0] = KernelPolicy::TILE_SIZE;
}



/******************************************************************************
 *
 *****************************************************************************/

/**
 * Structure reflecting KernelPolicy details
 */
template <typename KernelPtr>
struct FooKernelProperties : cub::KernelProperties<KernelPtr>
{
	typedef cub::KernelProperties<KernelPtr> Base;

	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	int tile_size;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	FooKernelProperties(KernelPtr kernel_ptr) :
		KernelProperties(kernel_ptr)
	{};

	/**
	 * Initializer
	 */
	template <typename KernelPolicy>
	cudaError_t Init(
		KernelPolicy policy,
		const cub::CudaProperties &cuda_props)
	{
		tile_size = KernelPolicy::TILE_SIZE;

		return Base::Init(
			KernelPolicy::TILE_SIZE,
			cuda_props);
	}

};


/******************************************************************************
 *
 *****************************************************************************/



/**
 * Foo dispatch assistant
 */
template <typename T, typename SizeT>
struct FooDispatch
{
	//---------------------------------------------------------------------
	// Tuned policy specializations
	//---------------------------------------------------------------------

	template <int TUNED_ARCH>
	struct TunedPolicy;

	// 100
	template <>
	struct TunedPolicy<100> : Policy<
		KernelPolicy<64, 1, 1>,
		KernelPolicy<64, 1, 1>,
		KernelPolicy<64, 1, 1>,
		true,
		true>
	{};

	// 200
	template <>
	struct TunedPolicy<200> : Policy<
		KernelPolicy<128, 2, 8>,
		KernelPolicy<128, 2, 8>,
		KernelPolicy<128, 2, 8>,
		true,
		true>
	{};

	// 300
	template <>
	struct TunedPolicy<300> : Policy<
		KernelPolicy<128, 4, 32>,
		KernelPolicy<128, 4, 32>,
		KernelPolicy<128, 4, 32>,
		false,
		false>
	{};


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

	// Tuning policies specific to the arch-id of the active compiler
	// pass.  (The policy's type signature is "opaque" to the target
	// architecture.)
	struct TunedUpsweep : 		TunedPolicy<TUNE_ARCH>::Upsweep {};
	struct TunedSpine : 		TunedPolicy<TUNE_ARCH>::Spine {};
	struct TunedDownsweep : 	TunedPolicy<TUNE_ARCH>::Downsweep {};

	// Type signatures of kernel entrypoints
	typedef void (*UpsweepKernelPtr)(T*, SizeT);
	typedef void (*SpineKernelPtr)(T*, SizeT);
	typedef void (*DownsweepKernelPtr)(T*, SizeT);



	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	cudaError_t									init_error;

	cub::CudaProperties							cuda_props;

	FooKernelProperties<UpsweepKernelPtr>		upsweep_props;
	FooKernelProperties<SpineKernelPtr>			spine_props;
	FooKernelProperties<DownsweepKernelPtr>		downsweep_props;

	// Host-specific tuning details
	bool 										uniform_smem_allocation;
	bool 										uniform_grid_size;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	// Initializer
	template <typename Policy>
	cudaError_t Init(Policy policy)
	{
		cudaError_t error = cudaSuccess;
		do {

			if (error = upsweep_props.Init(Policy::Upsweep(), cuda_props)) break;
			if (error = spine_props.Init(Policy::Spine(), cuda_props)) break;
			if (error = downsweep_props.Init(Policy::Downsweep(), cuda_props)) break;

			uniform_smem_allocation 	= Policy::UNIFORM_SMEM_ALLOCATION;
			uniform_grid_size 			= Policy::UNIFORM_GRID_SIZE;

		} while (0);

		return error;
	}


	/**
	 * Constructor.  Initializes kernel pointer and reflective fields using
	 * the supplied policy type.
	 */
	template <typename Policy>
	FooDispatch(Policy policy) :
		upsweep_props(FooKernel<Policy::Upsweep>),
		spine_props(FooKernel<Policy::Spine>),
		downsweep_props(FooKernel<Policy::Downsweep>)
	{
		do {
			if (init_error = cuda_props.init_error) break;
			if (init_error = Init(Policy())) break;
		} while (0);
	}


	/**
	 * Constructor.  Initializes kernel pointer and reflective fields using
	 * an appropriate policy specialization for the given ptx version.
	 */
	FooDispatch() :
		upsweep_props(FooKernel<TunedUpsweep>),
		spine_props(FooKernel<TunedSpine>),
		downsweep_props(FooKernel<TunedDownsweep>)
	{
		do {
			if (init_error = cuda_props.init_error) break;

			// Initialize kernel details with appropriate tuning parameterizations
			if (cuda_props.ptx_version >= 300) {

				if (init_error = Init(TunedPolicy<300>())) break;

			} else if (cuda_props.ptx_version >= 200) {

				if (init_error = Init(TunedPolicy<200>())) break;

			} else {

				if (init_error = Init(TunedPolicy<100>())) break;

			}
		} while (0);
	}


	/**
	 * Constructor.
	 */
	FooDispatch(
		FooKernelProperties<UpsweepKernelPtr>		upsweep_props,
		FooKernelProperties<SpineKernelPtr>			spine_props,
		FooKernelProperties<DownsweepKernelPtr>		downsweep_props,
		bool 										uniform_smem_allocation,
		bool 										uniform_grid_size) :
			upsweep_props(upsweep_props),
			spine_props(spine_props),
			downsweep_props(downsweep_props),
			uniform_smem_allocation(uniform_smem_allocation),
			uniform_grid_size(uniform_grid_size)
	{}


	/**
	 * Enact a Foo pass
	 */
	cudaError_t Enact(T* d_data, SizeT num_elements)
	{
		if (init_error) {
			return init_error;
		}

		cudaError_t retval = cudaSuccess;

		do {
			// Upsweep
			upsweep_props.kernel_ptr<<<1,1>>>(d_data, num_elements);
			if (retval = cudaDeviceSynchronize()) break;

			// Spine
			spine_props.kernel_ptr<<<1,1>>>(d_data, num_elements);
			if (retval = cudaDeviceSynchronize()) break;

			// Downsweep
			downsweep_props.kernel_ptr<<<1,1>>>(d_data, num_elements);
			if (retval = cudaDeviceSynchronize()) break;

		} while (0);

		return retval;
	}
};




/**
 *
 */
template <typename T, typename SizeT>
cudaError_t Foo(T* d_data, SizeT num_elements)
{
	FooDispatch<T, SizeT> dispatch;
	return dispatch.Enact(d_data, num_elements);
}


/**
 *
 */
template <typename Policy, typename T, typename SizeT>
cudaError_t Foo(Policy policy, T* d_data, SizeT num_elements)
{
	FooDispatch<T, SizeT> dispatch(policy);
	return dispatch.Enact(d_data, num_elements);
}






/******************************************************************************
 * Host Program
 *****************************************************************************/


/**
 * Test selection and dispatch of autotuned policy
 */
template <typename T, typename SizeT>
void TestAutotunedPolicy(T *d_data, SizeT num_elements)
{
	cudaError_t error;

	// Dispatch kernel
	Foo(d_data, num_elements);

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
template <typename T, typename SizeT>
void TestCustomPolicy(T *d_data, SizeT num_elements)
{
	cudaError_t error;

	// Declare custom policy
	Policy<
		KernelPolicy<256, 2, 1>,
		KernelPolicy<192, 2, 1>,
		KernelPolicy<384, 2, 1>,
		false,
		false> custom_policy;

	// Dispatch kernel
	Foo(custom_policy, d_data, num_elements);

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

	int num_elements = 3;

	// Test autotuned policy
	TestAutotunedPolicy(d_data, num_elements);

	// Test custom policy
	TestCustomPolicy(d_data, num_elements);

	return 0;
}
