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

#include "../../radix_sort/block/cta.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace block {


/**
 *
 */
template <
	typename KernelPolicy,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (KernelPolicy::CTA_THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__ 
void Kernel(
	Partition							*d_partitions_in,
	Partition							*d_partitions_out,
	KeyType 							*d_keys_in,
	KeyType 							*d_keys_out,
	KeyType 							*d_keys_final,
	ValueType 							*d_values_in,
	ValueType 							*d_values_out,
	ValueType 							*d_values_final,
	int									low_bit)
{
	// CTA abstraction type
	typedef Cta<KernelPolicy, KeyType, ValueType> Cta;

	// Shared memory pool
	__shared__ typename Cta::SmemStorage smem_storage;

	Cta cta(
		smem_storage,
		d_partitions_in,
		d_partitions_out,
		d_keys_in,
		d_keys_out,
		d_keys_final,
		d_values_in,
		d_values_out,
		d_values_final,
		low_bit);

	cta.ProcessWorkRange();
}



} // namespace block
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
