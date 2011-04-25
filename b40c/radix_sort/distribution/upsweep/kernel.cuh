/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Distribution sort upsweep digit-reduction/counting kernel.
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/radix_sort/distribution/upsweep/cta.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace upsweep {


/**
 * Upsweep reduction pass
 */
template <typename KernelConfig, typename SmemStorage>
__device__ __forceinline__ void UpsweepPass(
	int 								*&d_selectors,
	typename KernelConfig::SizeT 		*&d_spine,
	typename KernelConfig::KeyType 		*&d_in_keys,
	typename KernelConfig::KeyType 		*&d_out_keys,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition,
	SmemStorage							&smem_storage)
{
	typedef Cta<KernelConfig, SmemStorage> 	Cta;
	typedef typename KernelConfig::KeyType 			KeyType;
	typedef typename KernelConfig::SizeT 			SizeT;
	
	// Determine where to read our input

	bool selector = ((KernelConfig::EARLY_EXIT) && ((KernelConfig::CURRENT_PASS != 0) && (d_selectors[KernelConfig::CURRENT_PASS & 0x1]))) ||
		(KernelConfig::CURRENT_PASS & 0x1);
	KeyType *d_keys = (selector) ? d_out_keys : d_in_keys;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelConfig::LOG_TILE_ELEMENTS,
		KernelConfig::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_keys,
		d_spine);
	
	// Accumulate digit counts for all tiles
	cta.ProcessTiles(work_limits.offset, work_limits.out_of_bounds);
}


/**
 * Upsweep reduction kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void UpsweepKernel(
	int 								*d_selectors,
	typename KernelConfig::SizeT 		*d_spine,
	typename KernelConfig::KeyType 		*d_in_keys,
	typename KernelConfig::KeyType 		*d_out_keys,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> work_decomposition)
{
	// Shared memory pool
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	UpsweepPass<KernelConfig>(
		d_selectors,
		d_spine,
		d_in_keys,
		d_out_keys,
		work_decomposition,
		smem_storage);
}


} // namespace upsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

