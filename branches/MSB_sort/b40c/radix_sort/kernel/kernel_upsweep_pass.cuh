/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
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

#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/**
 * Kernel entry point
 */
template <
	typename UpsweepCtaPolicy,
	typename SizeT,
	typename KeyType>
__launch_bounds__ (
	UpsweepCtaPolicy::CTA_THREADS,
	UpsweepCtaPolicy::MIN_CTA_OCCUPANCY)
__global__
void UpsweepKernel(
	SizeT 								*d_spine,
	KeyType 							*d_in_keys,
	util::CtaWorkDistribution<SizeT> 	cta_work_distribution,
	unsigned int 						current_bit)
{
	// CTA abstraction type
	typedef UpsweepCta<UpsweepCtaPolicy, SizeT, KeyType> UpsweepCta;

	// Shared data structures
	__shared__ typename UpsweepCta::SmemStorage 	smem_storage;
	__shared__ util::CtaProgress<SizeT, TILE_ELEMENTS> 	cta_progress;


	// Determine our threadblock's work range
	if (threadIdx.x == 0)
	{
		cta_progress.Init(cta_work_distribution);
	}

	// Sync to acquire work range
	__syncthreads();

	// Compute bin-count for each radix digit (valid in tid < RADIX_DIGITS)
	SizeT bin_count;
	UpsweepCta::ProcessWorkRange(
		smem_storage,
		d_in_keys,
		current_bit,
		cta_progress.cta_offset,
		cta_progress.out_of_bounds,
		bin_count);

	// Write out the bin_count reductions
	if (threadIdx.x < RADIX_DIGITS)
	{
		int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;

		util::io::ModifiedStore<UpsweepCtaPolicy::STORE_MODIFIER>::St(
			bin_count,
			d_spine + spine_bin_offset);
	}
}


/**
 * Upsweep kernel properties
 */
struct UpsweepKernelProps : util::KernelProps
{
	// Kernel function type
	typedef void (*KernelFunc)(
		SizeT*,
		KeyType*,
		util::CtaWorkDistribution<SizeT>,
		unsigned int);

	// Fields
	KernelFunc 					kernel_func;
	int 						tile_elements;
	cudaSharedMemConfig 		sm_bank_config;

	/**
	 * Initializer
	 */
	template <
		typename KernelPolicy,
		typename OpaquePolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		// Initialize fields
		kernel_func 			= upsweep::Kernel<OpaquePolicy>;
		tile_elements 			= KernelPolicy::TILE_ELEMENTS;
		sm_bank_config 			= KernelPolicy::SMEM_CONFIG;

		// Initialize super class
		return util::KernelProps::Init(
			kernel_func,
			KernelPolicy::CTA_THREADS,
			sm_arch,
			sm_count);
	}

	/**
	 * Initializer
	 */
	template <typename KernelPolicy>
	cudaError_t Init(int sm_arch, int sm_count)
	{
		return Init<KernelPolicy, KernelPolicy>(sm_arch, sm_count);
	}
};


} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
