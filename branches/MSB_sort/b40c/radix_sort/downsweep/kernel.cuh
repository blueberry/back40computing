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
 * Radix sort downsweep scan kernel (scatter into bins)
 ******************************************************************************/

#pragma once

#include "../../radix_sort/downsweep/cta.cuh"
#include "../../util/cta_progress.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Radix sort downsweep scan kernel entry point
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (KernelPolicy::CTA_THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__ 
void Kernel(
	Partition							*d_partitions_out,
	SizeT 								*d_spine,
	KeyType 							*d_in_keys,
	KeyType 							*d_out_keys,
	ValueType 							*d_in_values,
	ValueType 							*d_out_values,
	util::CtaWorkDistribution<SizeT> 	cta_work_distribution,
	unsigned int 						current_bit)
{
	// CTA abstraction type
	typedef Cta<KernelPolicy, SizeT, KeyType, ValueType> Cta;

	// Shared memory pool
	__shared__ typename Cta::SmemStorage smem_storage;

	Cta cta(
		smem_storage,
		d_partitions_out,
		d_spine,
		d_in_keys,
		d_out_keys,
		d_in_values,
		d_out_values,
		current_bit);

	cta.ProcessWorkRange(cta_work_distribution);
}



} // namespace downsweep
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
