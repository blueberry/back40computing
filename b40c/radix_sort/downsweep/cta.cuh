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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/******************************************************************************
 * CTA-processing functionality for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/partition/downsweep/cta.cuh>
#include <b40c/radix_sort/downsweep/tile.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Radix sort downsweep scan CTA
 *
 * Derives from partition::downsweep::Cta
 */
template <
	typename KernelPolicy,
	bool FLOP_TURN>
struct Cta :
	partition::downsweep::Cta<
		KernelPolicy,
		FLOP_TURN,
		Cta<KernelPolicy, FLOP_TURN>,		// This class
		Tile>								// radix_sort::downsweep::Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	// Base class type
	typedef partition::downsweep::Cta<KernelPolicy, FLOP_TURN, Cta, Tile> Base;

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys0,
		KeyType 		*d_keys1,
		ValueType 		*d_values0,
		ValueType 		*d_values1,
		SizeT 			*d_spine) :
			Base(
				smem_storage,
				d_keys0,
				d_keys1,
				d_values0,
				d_values1,
				d_spine)
	{}
};


} // namespace downsweep
} // namespace radix_sort
} // namespace b40c

