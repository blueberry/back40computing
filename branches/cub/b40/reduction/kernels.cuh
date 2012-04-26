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
 * Reduction kernel tuning policy and entry-points
 ******************************************************************************/

#pragma once

#include <cub/cta_work_distribution.cuh>
#include <b40/reduction/cta.cuh>

namespace b40 {
namespace reduction {

using namespace cub;


/**
 * Upsweep reduction kernel entry point
 */
template <
	typename KernelPolicy,
	typename InputIterator,
	typename OutputIterator,
	typename ReductionOp,
	typename SizeT>
__global__ void UpsweepKernel(
	InputIterator					d_in,
	OutputIterator			 		d_out,
	ReductionOp						reduction_op,
	CtaWorkDistribution<SizeT> 		work_distribution)
{
	typedef Cta<
		KernelPolicy,
		InputIterator,
		OutputIterator,
		ReductionOp,
		SizeT> Cta;

	__shared__ typename Cta::SmemStorage smem_storage;

	Cta cta(smem_storage, d_in, d_out, reduction_op, work_distribution);
	cta.ProcessTiles();
}


/**
 * Single-CTA reduction kernel entry point
 */
template <
	typename KernelPolicy,
	typename InputIterator,
	typename OutputIterator,
	typename ReductionOp,
	typename SizeT>
__global__ void SingleKernel(
	InputIterator					d_in,
	OutputIterator			 		d_out,
	ReductionOp						reduction_op,
	SizeT							num_elements)
{
	typedef Cta<
		KernelPolicy,
		InputIterator,
		OutputIterator,
		ReductionOp,
		SizeT> Cta;

	__shared__ typename Cta::SmemStorage smem_storage;

	Cta cta(smem_storage, d_in, d_out, reduction_op, num_elements);
	cta.ProcessTiles();
}


} // namespace reduction
} // namespace b40

