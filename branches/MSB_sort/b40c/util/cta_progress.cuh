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
 * CTA Work management.
 *
 * A given CTA may receive one of three different amounts of
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload
 * does the extra work.
 *
 ******************************************************************************/

#pragma once

#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {


/**
 * Description of work distribution amongst CTAs
 */
template <typename SizeT>
struct CtaWorkDistribution
{
	int total_grains;
	int overfetch;

	int big_blocks;
	int last_block;

	SizeT unguarded_elements;
	SizeT big_share;
	SizeT normal_share;
	SizeT normal_base_offset;


	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ CtaWorkDistribution(
		SizeT num_elements,
		int grid_size,
		int schedule_granularity)
	{
		total_grains 			= (num_elements + schedule_granularity - 1) / schedule_granularity;
		overfetch 				= (total_grains * schedule_granularity) - num_elements;

		int grains_per_cta 		= total_grains / grid_size;
		big_blocks	 			= total_grains - (grains_per_cta * grid_size);		// leftover grains go to big blocks

		normal_share 			= grains_per_cta * schedule_granularity;
		normal_base_offset 		= big_blocks * schedule_granularity;
		big_share 				= normal_share + schedule_granularity;

		last_block 				= CUB_MIN(total_grains, grid_size) - 1;
	}

	/**
	 * Print to stdout
	 */
	void Print()
	{
		printf(
			"total_grains: %lu, "
			"big_blocks: %lu, "
			"unguarded_elements: %lu, "
			"big_share: %lu, "
			"normal_share: %lu, "
			"normal_base_offset: %lu, "
			"last_block: %lu, "
			"overfetch: %lu \n",
				(unsigned long) total_grains,
				(unsigned long) big_blocks,
				(unsigned long) unguarded_elements,
				(unsigned long) big_share,
				(unsigned long) normal_share,
				(unsigned long) normal_base_offset,
				(unsigned long) last_block,
				(unsigned long) overfetch);
	}
};



/**
 *
 */
template <
	typename 	SizeT,
	int 		TILE_ELEMENTS,
	bool 		WORK_STEALING = false>
struct CtaProgress
{
	// Even share parameters
	SizeT 	cta_offset;
	SizeT 	out_of_bounds;

	/**
	 * Initializer
	 */
	__device__ __forceinline__ void Init(
		const CtaWorkDistribution<SizeT> &distribution)
	{

		if (WORK_STEALING) {

			// This CTA gets at least one tile (if possible)
			// TODO

		} else if (blockIdx.x < distribution.big_blocks) {

			// This CTA gets a big share of grains (grains_per_cta + 1)
			cta_offset = (blockIdx.x * distribution.big_share);
			out_of_bounds = cta_offset + distribution.big_share;

		} else if (blockIdx.x < distribution.total_grains) {

			// This CTA gets a normal share of grains (grains_per_cta)
			cta_offset = distribution.normal_base_offset + (blockIdx.x * distribution.normal_share);
			out_of_bounds = cta_offset + distribution.normal_share;

		} else {

			// This CTA gets no work
			cta_offset = 0;
			out_of_bounds = 0;
		}

		if (blockIdx.x == distribution.last_block)
		{
			out_of_bounds -= distribution.overfetch;
		}
	}


	/**
	 * Initializer
	 */
	__device__ __forceinline__ void Init(SizeT num_elements)
	{
		cta_offset = 0;
		out_of_bounds = num_elements;
	}


	/**
	 *
	 */
	__device__ __forceinline__ bool HasTile()
	{
		return (cta_offset < out_of_bounds);
	}


	/**
	 *
	 */
	__device__ __forceinline__ void NextTile()
	{
		if (WORK_STEALING) {

			// TODO

		} else {

			cta_offset += TILE_ELEMENTS;
		}
	}

};


} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
