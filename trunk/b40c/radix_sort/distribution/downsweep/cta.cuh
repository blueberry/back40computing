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
 * Upsweep CTA tile processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>
#include <b40c/util/device_intrinsics.cuh>

#include <b40c/radix_sort/sort_utils.cuh>
#include <b40c/radix_sort/distribution/downsweep/tile.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace downsweep {


/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <
	typename KernelConfig,
	typename SmemStorage,
	typename Derived = util::NullType>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelConfig::KeyType 					KeyType;
	typedef typename KernelConfig::ValueType 				ValueType;
	typedef typename KernelConfig::SizeT 					SizeT;
	typedef typename KernelConfig::Grid::LanePartial		LanePartial;

	typedef typename util::If<util::Equals<Derived, util::NullType>::VALUE, Cta, Derived>::Type Dispatch;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelConfig::SmemStorage 	&smem_storage;

	// Input and output device pointers
	KeyType								*d_in_keys;
	KeyType								*d_out_keys;

	ValueType							*d_in_values;
	ValueType							*d_out_values;

	SizeT								*d_spine;

	// SRTS details
	LanePartial							base_partial;
	int									*raking_segment;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		KeyType 		*d_out_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		SizeT 			*d_spine,
		LanePartial		base_partial,
		int				*raking_segment) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_out_keys(d_out_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			d_spine(d_spine),
			base_partial(base_partial),
			raking_segment(raking_segment)
	{
		if (threadIdx.x < KernelConfig::RADIX_DIGITS) {

			// Reset value-area of digit_warpscan
			smem_storage.digit_warpscan[1][threadIdx.x] = 0;

			// Read digit_carry in parallel
			SizeT my_digit_carry;
			int spine_digit_offset = util::FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
			util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
				my_digit_carry,
				d_spine + spine_digit_offset);
			smem_storage.digit_carry[threadIdx.x] = my_digit_carry;
		}
	}

	/**
	 * DecodeDigit
	 */
	__device__ __forceinline__ int DecodeDigit(KeyType key)
	{
		int digit;
		radix_sort::ExtractKeyBits<
			KeyType,
			KernelConfig::CURRENT_BIT,
			KernelConfig::RADIX_BITS>::Extract(digit, key);
		return digit;
	}

	/**
	 * Process
	 */
	__device__ __forceinline__ void ProcessTiles()
	{
		SizeT cta_offset = smem_storage.work_limits.offset;

		Tile<KernelConfig> tile;
		while (cta_offset < smem_storage.work_limits.guarded_offset) {
			tile.Process<true>(
				this,
				cta_offset,
				0);

			cta_offset += KernelConfig::TILE_ELEMENTS;
		}

		if (smem_storage.work_limits.guarded_elements) {
			tile.Process<false>(
				this,
				cta_offset,
				smem_storage.work_limits.guarded_elements);
		}
	}
};


} // namespace downsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

