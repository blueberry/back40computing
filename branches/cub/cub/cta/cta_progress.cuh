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
 * CTA Work management.
 *
 * A given CTA may receive one of three different amounts of
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
struct WorkDistribution
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
	__host__ __device__ __forceinline__ WorkDistribution(
		SizeT num_elements,
		int grid_size,
		int schedule_granularity)
	{
		total_grains 			= num_elements / schedule_granularity;
		unguarded_elements 		= total_grains * schedule_granularity;
		extra_elements 			= num_elements - unguarded_elements;

		int grains_per_cta 		= total_grains / grid_size;
		extra_grains 			= total_grains - (grains_per_cta * grid_size);

		normal_share 			= grains_per_cta * schedule_granularity;
		normal_base_offset 		= extra_grains * schedule_granularity;
		big_share 				= normal_share + schedule_granularity;

		last_block 				= CUB_MIN(total_grains, grid_size) - 1;
	}
};



/**
 *
 */
template <
	typename 	SizeT,
	int 		TILE_ELEMENTS,
	bool 		WORK_STEALING>
struct CtaProgress
{
	// Even share parameters
	SizeT 	cta_offset;
	SizeT 	out_of_bounds;

	int		extra_elements;
	SizeT	extra_offset;

	/**
	 * Constructor
	 */
	__device__ __forceinline__ CtaProgress(
		const WorkDistribution<SizeT> &distribution)
	{
		extra_elements = (blockIdx.x == distribution.last_block) ?
			distribution.extra_elements :
			0;

		if (WORK_STEALING) {

			// This CTA gets at least one tile (if possible)
			cta_offset = blockIdx.x * TILE_ELEMENTS;
			out_of_bounds = distribution.unguarded_elements;

		} else if (blockIdx.x < distribution.extra_grains) {

			// This CTA gets a big share (grains_per_cta + 1 grains)
			cta_offset = (blockIdx.x * distribution.big_share);
			out_of_bounds = cta_offset + distribution.big_share;

		} else if (blockIdx.x < distribution.total_grains) {

			// This CTA gets a normal share (grains_per_cta grains)
			cta_offset = distribution.normal_base_offset + (blockIdx.x * distribution.normal_share);
			out_of_bounds = cta_offset + distribution.normal_share;

		} else {

			// This CTA gets no work
			cta_offset = 0;
			out_of_bounds = 0;
		}

		extra_offset = out_of_bounds;
	}


	/**
	 * Constructor
	 */
	__device__ __forceinline__ CtaProgress(SizeT num_elements)
	{
		cta_offset = 0;
		extra_elements = num_elements % TILE_ELEMENTS;
		out_of_bounds = num_elements - extra_elements;
		extra_offset = out_of_bounds;
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

		} else {

			cta_offset += TILE_ELEMENTS;
		}
	}

};


} // namespace cub

