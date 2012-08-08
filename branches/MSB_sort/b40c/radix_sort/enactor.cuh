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
 * Radix sorting enactor
 ******************************************************************************/

#pragma once

#include "../radix_sort/problem_instance.cuh"
#include "../radix_sort/pass_policy.cuh"
#include "../util/error_utils.cuh"
#include "../util/scratch.cuh"
#include "../util/cuda_properties.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/**
 * Radix sorting enactor class
 */
struct Enactor
{
	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Scratch spine;

	// Pair of partition descriptor queues
	util::Scratch partitions[2];

	// Device properties
	const util::CudaProperties cuda_props;


	//---------------------------------------------------------------------
	// Helper structures
	//---------------------------------------------------------------------

	/**
	 * Tuned pass policy whose type signature does not reflect the tuned
	 * SM architecture.
	 */
	template <
		typename 		ProblemInstance,
		ProblemSize 	PROBLEM_SIZE,
		int 			RADIX_BITS>
	struct OpaquePassPolicy
	{
		// The appropriate tuning arch-id from the arch-id targeted by the
		// active compiler pass.
		enum
		{
/*
			COMPILER_TUNE_ARCH 		= (__CUB_CUDA_ARCH__ >= 200) ?
										200 :
										(__CUB_CUDA_ARCH__ >= 130) ?
											130 :
											100
*/
			COMPILER_TUNE_ARCH = 200
		};

		// Tuned pass policy
		typedef TunedPassPolicy<
			COMPILER_TUNE_ARCH,
			ProblemInstance,
			PROBLEM_SIZE,
			RADIX_BITS> TunedPassPolicy;

		struct DispatchPolicy 	: TunedPassPolicy::DispatchPolicy {};
		struct UpsweepPolicy 	: TunedPassPolicy::UpsweepPolicy {};
		struct SpinePolicy 		: TunedPassPolicy::SpinePolicy {};
		struct DownsweepPolicy 	: TunedPassPolicy::DownsweepPolicy {};
		struct PartitionPolicy 	: TunedPassPolicy::PartitionPolicy {};
		struct TilePolicy 		: TunedPassPolicy::TilePolicy {};
	};


	/**
	 * Sort.
	 */
	template <
		int 			TUNE_ARCH,
		ProblemSize 	PROBLEM_SIZE,
		typename 		ProblemInstance>
	cudaError_t Sort(ProblemInstance &problem_instance)
	{
		cudaError_t error = cudaSuccess;
		do
		{
			enum
			{
				RADIX_BITS = PreferredDigitBits<TUNE_ARCH>::PREFERRED_BITS,
			};

			// Define tuned and opaque pass policies
			typedef radix_sort::TunedPassPolicy<TUNE_ARCH, ProblemInstance, PROBLEM_SIZE, RADIX_BITS> 	TunedPassPolicy;
			typedef OpaquePassPolicy<ProblemInstance, PROBLEM_SIZE, RADIX_BITS>							OpaquePassPolicy;

			int sm_version = cuda_props.device_sm_version;
			int sm_count = cuda_props.device_props.multiProcessorCount;
			int initial_selector = problem_instance.storage.selector;

			// Upsweep kernel props
			typename ProblemInstance::UpsweepKernelProps upsweep_props;
			error = upsweep_props.template Init<
				typename TunedPassPolicy::UpsweepPolicy,
				typename OpaquePassPolicy::UpsweepPolicy>(sm_version, sm_count);
			if (error) break;

			// Spine kernel props
			typename ProblemInstance::SpineKernelProps spine_props;
			error = spine_props.template Init<
				typename TunedPassPolicy::SpinePolicy,
				typename OpaquePassPolicy::SpinePolicy>(sm_version, sm_count);
			if (error) break;

			// Downsweep kernel props
			typename ProblemInstance::DownsweepKernelProps downsweep_props;
			error = downsweep_props.template Init<
				typename TunedPassPolicy::DownsweepPolicy,
				typename OpaquePassPolicy::DownsweepPolicy>(sm_version, sm_count);
			if (error) break;

			// Partition kernel props
			typename ProblemInstance::PartitionKernelProps partition_props;
			error = partition_props.template Init<
				typename TunedPassPolicy::PartitionPolicy,
				typename OpaquePassPolicy::PartitionPolicy>(sm_version, sm_count);
			if (error) break;
/*
			// Tile kernel props
			typename ProblemInstance::TileKernelProps tile_props;
			error = tile_props.template Init<
				typename TunedPassPolicy::TilePolicy,
				typename OpaquePassPolicy::TilePolicy>(sm_version, sm_count);
			if (error) break;
*/
			//
			// Allocate
			//

			// Make sure our partition descriptor queues are big enough
			int max_partitions = (problem_instance.num_elements + partition_props.tile_elements - 1) / partition_props.tile_elements;
			size_t queue_bytes = sizeof(Partition) * max_partitions;

			error = partitions[0].Setup(queue_bytes);
			if (error) break;
			error = partitions[1].Setup(queue_bytes);
			if (error) break;

			error = cudaMemSet(partitions[0](), 0, sizeof(Partition) * max_partitions);
			if (util::B40CPerror(error, __FILE__, __LINE__)) break;
			error = cudaMemSet(partitions[1](), 0, sizeof(Partition) * max_partitions);
			if (util::B40CPerror(error, __FILE__, __LINE__)) break;


			//
			// First pass
			//

			// Print debug info
			if (problem_instance.debug)
			{
				printf("\nLow bit(%d), num bits(%d), radix_bits(%d), tuned arch(%d), SM arch(%d)\n",
					problem_instance.low_bit,
					problem_instance.num_bits,
					RADIX_BITS,
					TUNE_ARCH,
					cuda_props.device_sm_version);
				fflush(stdout);
			}

			// Dispatch first pass
			error = problem_instance.DispatchPrimary(
				RADIX_BITS,
				upsweep_props,
				spine_props,
				downsweep_props,
				TunedPassPolicy::DispatchPolicy::UNIFORM_GRID_SIZE,
				TunedPassPolicy::DispatchPolicy::DYNAMIC_SMEM_CONFIG);
			if (error) break;

			// Perform block iterations
			int grid_size = 32;
			{
				error = problem_instance.DispatchPartition(
					partition_props,
					initial_selector,
					grid_size);
				if (error) break;

				grid_size *= 32;
			}

			// Reset selector
			problem_instance.storage.selector = initial_selector;

		} while (0);

		return error;
	}



	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	Enactor()
	{}


	/**
	 * Enact a sort.
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::DoubleBuffer
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <ProblemSize PROBLEM_SIZE, typename DoubleBuffer>
	cudaError_t Sort(
		DoubleBuffer& 	problem_storage,
		int 			num_elements,
		int				low_bit,
		int 			num_bits,
		cudaStream_t	stream 			= 0,
		int 			max_grid_size 	= 0,
		bool 			debug 			= false)
	{
		typedef ProblemInstance<DoubleBuffer, int> ProblemInstance;

		if (num_elements <= 1)
		{
			// Nothing to do
			return cudaSuccess;
		}

		ProblemInstance problem_instance(
			problem_storage,
			num_elements,
			low_bit,
			num_bits,
			stream,
			spine,
			partitions,
			max_grid_size,
			debug);

//		if (cuda_props.kernel_ptx_version >= 200)
		{
			return Sort<200, PROBLEM_SIZE>(problem_instance);
		}
/*		else if (cuda_props.kernel_ptx_version >= 130)
		{
			return Sort<130, PROBLEM_SIZE>(problem_instance);
		}
		else
		{
			return Sort<100, PROBLEM_SIZE>(problem_instance);
		}
*/
	}

};





} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
