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
 * Radix sorting enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/radix_sort/problem_instance.cuh>
#include <b40c/radix_sort/tuned_pass_policy.cuh>

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
	util::Spine spine;

	// Device properties
	const util::CudaProperties cuda_props;


	//---------------------------------------------------------------------
	// Helper structures
	//---------------------------------------------------------------------

	/**
	 * Helper structure for iterating passes.
	 */
	template <
		typename 		ProblemInstance,
		ProblemSize 	PROBLEM_SIZE,
		int 			BITS_REMAINING,
		int 			CURRENT_BIT,
		int 			CURRENT_PASS>
	struct IteratePasses
	{
		// The appropriate tuning arch-id from the arch-id targeted by the
		// active compiler pass.
		enum {
			COMPILER_TUNE_ARCH = 200,
/*
			VALUE = (__CUB_CUDA_ARCH__ >= 200) ?
				200 :
				(__CUB_CUDA_ARCH__ >= 130) ?
					130 :
					100
*/
		};

		typedef TunedPassPolicy<
			COMPILER_TUNE_ARCH,
			ProblemInstance,
			PROBLEM_SIZE,
			BITS_REMAINING,
			CURRENT_BIT,
			CURRENT_PASS> CompilerTunedPolicy;

		struct OpaqueUpsweepPolicy 		: CompilerTunedPolicy::UpsweepPolicy {};
		struct OpaqueSpinePolicy 		: CompilerTunedPolicy::SpinePolicy {};
		struct OpaqueDownsweepPolicy 	: CompilerTunedPolicy::DownsweepPolicy {};

		/**
		 * DispatchPass pass
		 */
		template <int TUNE_ARCH>
		static cudaError_t DispatchPass(
			ProblemInstance &problem_instance,
			Enactor &enactor)
		{
			typedef TunedPassPolicy<
				TUNE_ARCH,
				ProblemInstance,
				PROBLEM_SIZE,
				BITS_REMAINING,
				CURRENT_BIT,
				CURRENT_PASS> TunedPolicy;

			typedef typename TunedPolicy::UpsweepPolicy 	TunedUpsweepPolicy;
			typedef typename TunedPolicy::SpinePolicy 		TunedSpinePolicy;
			typedef typename TunedPolicy::DownsweepPolicy 	TunedDownsweepPolicy;


			static const int RADIX_BITS = TunedPolicy::DispatchPolicy::RADIX_BITS;

			cudaError_t error = cudaSuccess;
			do {
				if (problem_instance.debug) {
					printf("\nCurrent bit(%d), Pass(%d), Radix bits(%d), tuned arch(%d), SM arch(%d)\n",
						CURRENT_BIT, CURRENT_PASS, RADIX_BITS, TUNE_ARCH, enactor.cuda_props.device_sm_version);
					fflush(stdout);
				}

				// Upsweep kernel props
				typename ProblemInstance::UpsweepKernelProps upsweep_props;
				error = upsweep_props.template Init<TunedUpsweepPolicy, OpaqueUpsweepPolicy>(
					enactor.cuda_props.device_sm_version,
					enactor.cuda_props.device_props.multiProcessorCount);
				if (error) break;

				// Spine kernel props
				typename ProblemInstance::SpineKernelProps spine_props;
				error = spine_props.template Init<TunedSpinePolicy, OpaqueSpinePolicy>(
					enactor.cuda_props.device_sm_version,
					enactor.cuda_props.device_props.multiProcessorCount);
				if (error) break;

				// Downsweep kernel props
				typename ProblemInstance::DownsweepKernelProps downsweep_props;
				error = downsweep_props.template Init<TunedDownsweepPolicy, OpaqueDownsweepPolicy>(
					enactor.cuda_props.device_sm_version,
					enactor.cuda_props.device_props.multiProcessorCount);
				if (error) break;

				// Dispatch current pass
				error = problem_instance.DispatchPass(
					RADIX_BITS,
					upsweep_props,
					spine_props,
					downsweep_props,
					TunedPolicy::DispatchPolicy::UNIFORM_GRID_SIZE,
					TunedPolicy::DispatchPolicy::UNIFORM_SMEM_ALLOCATION);
				if (error) break;

				// DispatchPass next pass
				error = IteratePasses<
					ProblemInstance,
					PROBLEM_SIZE,
					BITS_REMAINING - RADIX_BITS,
					CURRENT_BIT + RADIX_BITS,
					CURRENT_PASS + 1>::template DispatchPass<TUNE_ARCH>(problem_instance, enactor);
				if (error) break;

			} while (0);

			return error;
		}
	};


	/**
	 * Helper structure for iterating passes. (Termination)
	 */
	template <typename ProblemInstance, ProblemSize PROBLEM_SIZE, int CURRENT_BIT, int NUM_PASSES>
	struct IteratePasses<ProblemInstance, PROBLEM_SIZE, 0, CURRENT_BIT, NUM_PASSES>
	{
		/**
		 * DispatchPass pass
		 */
		template <int TUNE_ARCH>
		static cudaError_t DispatchPass(
			ProblemInstance &problem_instance,
			Enactor &enactor)
		{
			// We moved data between storage buffers at every pass
			problem_instance.storage.selector =
				(problem_instance.storage.selector + NUM_PASSES) & 0x1;

			return cudaSuccess;
		}
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	Enactor() {}


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
	template <
		ProblemSize PROBLEM_SIZE,
		int BITS_REMAINING,
		int CURRENT_BIT,
		typename DoubleBuffer>
	cudaError_t Sort(
		DoubleBuffer& 	problem_storage,
		int 			num_elements,
		cudaStream_t	stream 			= 0,
		int 			max_grid_size 	= 0,
		bool 			debug 			= false)
	{
		typedef ProblemInstance<DoubleBuffer, int> ProblemInstance;

		ProblemInstance problem_instance(
			problem_storage,
			num_elements,
			stream,
			spine,
			max_grid_size,
			debug);

/*
		if (cuda_props.kernel_ptx_version >= 200)
		{
			return IteratePasses<ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT, 0>::
				template DispatchPass<200>(problem_instance, *this);
		}
		else if (cuda_props.kernel_ptx_version >= 130)
		{
			return IteratePasses<ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT, 0>::
				template DispatchPass<130>(problem_instance, *this);
		}
		else
		{
			return IteratePasses<ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT, 0>::
				template DispatchPass<100>(problem_instance, *this);
		}
*/
		return IteratePasses<ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT, 0>::
			template DispatchPass<200>(problem_instance, *this);

	}
};





} // namespace radix_sort
} // namespace b40c

