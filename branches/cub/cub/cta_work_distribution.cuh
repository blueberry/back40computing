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
 * Work management data structures.
 *
 * A given threadblock may receive one of three different amounts of
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload
 * does the extra work.
 *
 ******************************************************************************/

#pragma once

namespace cub {




/**
 * Description of work distribution amongst CTAs
 */
template <typename SizeT>
struct CtaWorkDistribution
{
	int total_grains;
	int extra_grains;

	SizeT unguarded_elements;
	SizeT big_share;
	SizeT normal_share;
	SizeT normal_base_offset;

	int last_block;
	int extra_elements;

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ void CtaWorkDistribution(
		SizeT num_elements,
		int grid_size,
		int schedule_granularity)
	{
		total_grains 			= num_elements / schedule_granularity;
		unguarded_elements 		= total_grains * schedule_granularity;
		extra_elements 			= num_elements - unguarded_elements;

		int grains_per_cta 		= total_grains / grid_size;													// round down for the ks
		extra_grains 			= total_grains - (grains_per_cta * grid_size);

		normal_share 			= grains_per_cta * schedule_granularity;
		normal_base_offset 		= extra_grains * schedule_granularity;
		big_share 				= normal.elements + schedule_granularity;

		last_block 				= CUB_MIN(total_grains, grid_size) - 1;
	}
};




template <
	typename SizeT,
	int TILE_ELEMENTS,
	bool WORK_STEAL>
struct CtaProgress
{
	// Even share parameters
	SizeT 	cta_base;
	SizeT 	out_of_bounds;
	int		extra_elements;

	/**
	 * Constructor
	 */
	__device__ __forceinline__ CtaProgress(
		const CtaWorkDistribution<SizeT> &distribution)
	{
		extra_elements = distribution.extra_elements;

		if (WORK_STEAL) {

			// This CTA gets at least one tile (if possible)
			cta_base = blockIdx.x * TILE_ELEMENTS;
			out_of_bounds = distribution.unguarded_elements;

		} else if (blockIdx.x < distribution.extra_grains) {

			// This CTA gets a big share (grains_per_cta + 1 grains)
			cta_base = (blockIdx.x * distribution.big_share);
			out_of_bounds = cta_base + distribution.big_share;

		} else if (blockIdx.x < distribution.total_grains) {

			// This CTA gets a normal share (grains_per_cta grains)
			cta_base = distribution.normal_base_offset + (blockIdx.x * distribution.normal_share);
			out_of_bounds = cta_base + distribution.normal_share;

		} else {

			// This CTA gets no work
			cta_base = 0;
			out_of_bounds = 0;
		}
	}


	/**
	 *
	 */
	__device__ __forceinline__ bool FirstTile(SizeT &cta_offset )
	{
		cta_offset = this->cta_base;
		this->cta_base += TILE_ELEMENTS;
		return (cta_offset < out_of_bounds);
	}


	/**
	 *
	 */
	__device__ __forceinline__ bool NextTile(SizeT &cta_offset)
	{
		if (WORK_STEAL) {

		} else {

			cta_offset = this->cta_base;
			this->cta_base += TILE_ELEMENTS;
		}
		return (cta_offset < out_of_bounds);
	}


	/**
	 *
	 */
	__device__ __forceinline__ bool LastPartialTile(SizeT &cta_offset, int &num_valid)
	{
		cta_offset = out_of_bounds;
		num_valid = extra_elements;

		return (blockIdx.x == distribution.last_block);
	}


};


} // namespace cub

