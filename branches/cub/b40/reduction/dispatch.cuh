/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 *
 ******************************************************************************/

#pragma once

#include <cub/core/cuda_properties.cuh>
#include <cub/dispatch/kernel_properties.cuh>
#include <cub/dispatch/cta_work_distribution.cuh>
#include <b40/reduction/policy.cuh>
#include <b40/reduction/kernels.cuh>


namespace b40 {
namespace reduction {

using namespace cub;




/**
 * Reduction dispatch assistant
 */
template <
	typename InputIterator,
	typename OutputIterator,
	typename SizeT,
	typename ReductionOp>
struct Dispatch
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	typedef typename std::iterator_traits<InputIterator>::value_type T;

	// Type signatures of kernel entrypoints
	typedef void (*UpsweepKernelPtr)	(InputIterator, T*, ReductionOp, CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)		(T*, OutputIterator, ReductionOp, SizeT);
	typedef void (*SingleKernelPtr)		(InputIterator, OutputIterator, ReductionOp, SizeT);


	/**
	 * Structure reflecting KernelPolicy details
	 */
	template <typename KernelPtr>
	struct KernelDetails : KernelProperties<KernelPtr>
	{
		int tile_elements;

		/**
		 * Initializer
		 */
		template <typename KernelPolicy>
		cudaError_t Init(
			KernelPolicy policy,
			KernelPtr kernel_ptr,
			const CudaProperties &cuda_props)
		{
			this->tile_elements = KernelPolicy::TILE_ELEMENTS;

			return KernelProperties<KernelPtr>::Init(
				kernel_ptr,
				KernelPolicy::THREADS,
				cuda_props);
		}
	};


	//---------------------------------------------------------------------
	// Tuned policy specializations
	//---------------------------------------------------------------------

	template <int TUNED_ARCH>
	struct TunedPolicy;

	// 100
	template <>
	struct TunedPolicy<100> : Policy<
		KernelPolicy<64, 1, 1, READ_NONE, WRITE_NONE, false>,
		KernelPolicy<64, 1, 1, READ_NONE, WRITE_NONE, false>,
		true,
		true>
	{};

	// 130
	template <>
	struct TunedPolicy<130> : Policy<
		KernelPolicy<128, 1, 2, READ_NONE, WRITE_NONE, false>,
		KernelPolicy<128, 1, 2, READ_NONE, WRITE_NONE, false>,
		true,
		true>
	{};

	// 200
	template <>
	struct TunedPolicy<200> : Policy<
		KernelPolicy<128, 2, 2, READ_NONE, WRITE_NONE, false>,
		KernelPolicy<128, 2, 2, READ_NONE, WRITE_NONE, false>,
		true,
		true>
	{};



	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	// Determine the appropriate tuning arch-id from the arch-id targeted
	// by the active compiler pass.
	enum {
		TUNE_ARCH =
			(__CUB_CUDA_ARCH__ >= 200) ?
				200 :
				(__CUB_CUDA_ARCH__ >= 130) ?
					130 :
					100,
	};

	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	// Tuning policies specific to the arch-id of the active compiler
	// pass.  (The policy's type signature is "opaque" to the target
	// architecture.)
	struct TunedUpsweep : 		TunedPolicy<TUNE_ARCH>::Upsweep {};
	struct TunedSingle : 		TunedPolicy<TUNE_ARCH>::Single {};



	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	cudaError_t							init_error;

	CudaProperties						cuda_props;

	KernelDetails<UpsweepKernelPtr>		upsweep_props;
	KernelDetails<SpineKernelPtr>		spine_props;
	KernelDetails<SingleKernelPtr>		single_props;

	// Host-specific tuning details
	bool 								uniform_smem_allocation;
	bool 								uniform_grid_size;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	// Initializer
	template <typename Policy>
	cudaError_t Init(
		Policy 				policy,
		UpsweepKernelPtr	upsweep_ptr 	= UpsweepKernel<TunedUpsweep>,
		SpineKernelPtr		spine_ptr 		= SingleKernel<TunedSingle>,
		SingleKernelPtr		single_ptr 		= SingleKernel<TunedSingle>)
	{
		cudaError_t error = cudaSuccess;
		do {

			if (error = upsweep_props.Init(Policy::Upsweep(), upsweep_ptr, cuda_props)) break;
			if (error = spine_props.Init(Policy::Single(), spine_ptr, cuda_props)) break;
			if (error = single_props.Init(Policy::Single(), single_ptr, cuda_props)) break;

			uniform_smem_allocation 	= Policy::UNIFORM_SMEM_ALLOCATION;
			uniform_grid_size 			= Policy::UNIFORM_GRID_SIZE;

		} while (0);

		return error;
	}


	/**
	 * Constructor.  Initializes kernel pointers and reflective fields using
	 * the supplied policy type.
	 */
	template <typename Policy>
	Dispatch(Policy policy)
	{
		do {
			if (init_error = cuda_props.init_error) break;

			if (init_error = Init(
				Policy(),
				UpsweepKernel<Policy::Upsweep>,
				SingleKernel<Policy::Single>,
				SingleKernel<Policy::Single>)) break;

		} while (0);
	}


	/**
	 * Constructor.  Initializes kernel pointers and reflective fields using
	 * an appropriate policy specialization for the given ptx version.
	 */
	Dispatch()
	{
		do {
			if (init_error = cuda_props.init_error) break;

			// Initialize kernel details with appropriate tuning parameterizations
			if (cuda_props.ptx_version >= 200) {

				if (init_error = Init(TunedPolicy<200>())) break;

			} else if (cuda_props.ptx_version >= 130) {

				if (init_error = Init(TunedPolicy<130>())) break;

			} else {

				if (init_error = Init(TunedPolicy<100>())) break;

			}
		} while (0);
	}


	/**
	 * Constructor.
	 */
	Dispatch(
		KernelDetails<UpsweepKernelPtr>			upsweep_props,
		KernelDetails<SpineKernelPtr>			spine_props,
		KernelDetails<SingleKernelPtr>			single_props,
		bool 										uniform_smem_allocation,
		bool 										uniform_grid_size) :
			upsweep_props(upsweep_props),
			spine_props(spine_props),
			single_props(single_props),
			uniform_smem_allocation(uniform_smem_allocation),
			uniform_grid_size(uniform_grid_size)
	{}


	/**
	 * Enact a reduction pass
	 */
	cudaError_t Enact(
		InputIterator 		first,
		OutputIterator 		result,
		SizeT 				num_elements,
		ReductionOp 		reduction_op,
		int 				max_grid_size = 0)
	{
		if (init_error) {
			return init_error;
		}

		cudaError_t retval = cudaSuccess;
		do {

			// Dispatch kernels

		} while (0);

		return retval;
	}
};




}// namespace reduction
}// namespace b40

