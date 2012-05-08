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
#include <b40c/radix_sort/problem_instance.cuh>

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
		typename 		CompilerTuneArch,
		typename 		ProblemInstance,
		ProblemSize 	PROBLEM_SIZE,
		int 			BITS_REMAINING,
		int 			CURRENT_BIT,
		int 			CURRENT_PASS>
	struct IteratePasses
	{
		typedef TunedPassPolicy<
			CompilerTuneArch::VALUE,
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
		static cudaError_t DispatchPass(ProblemInstance &problem_instance)
		{
			typedef TunedPassPolicy<
				TUNE_ARCH,
				ProblemInstance,
				PROBLEM_SIZE,
				BITS_REMAINING,
				CURRENT_BIT,
				CURRENT_PASS> TunedPolicy;

			static const int RADIX_BITS = TunedPolicy::DispatchPolicy::RADIX_BITS;

			cudaError_t error = cudaSuccess;
			do {
				if (debug) {
					printf("\nCurrent bit(%d), Pass(%d), Radix bits(%d), tuned arch(%d), SM arch(%d)\n",
						CURRENT_BIT, CURRENT_PASS, RADIX_BITS, TUNE_ARCH, cuda_props.device_sm_version);
					fflush(stdout);
				}

				ProblemInstance::UpsweepKernelProps upsweep_props;
				error = upsweep_props.Init<TunedPolicy::UpsweepPolicy, OpaqueUpsweepPolicy>(
					cuda_props.device_sm_version,
					cuda_props.device_props.cuda_props.device_props.multiProcessorCount);
				if (error) break;

				ProblemInstance::SpineKernelProps spine_props;
				error = spine_props.Init<TunedPolicy::SpinePolicy, OpaqueSpinePolicy>(
					cuda_props.device_sm_version,
					cuda_props.device_props.cuda_props.device_props.multiProcessorCount);
				if (error) break;

				ProblemInstance::DownsweepKernelProps downsweep_props;
				error = downsweep_props.Init<TunedPolicy::DownsweepPolicy, OpaqueDownsweepPolicy>(
					cuda_props.device_sm_version,
					cuda_props.device_props.cuda_props.device_props.multiProcessorCount);
				if (error) break;

				// DispatchPass current pass
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
					TUNE_ARCH,
					BITS_REMAINING - RADIX_BITS,
					CURRENT_BIT + RADIX_BITS,
					CURRENT_PASS + 1>::template DispatchPass<TUNE_ARCH>(problem_instance);
				if (error) break;

			} while (0);

			return error;
		}
	};


	/**
	 * Helper structure for iterating passes. (Termination)
	 */
	template <typename ProblemInstance, typename CompilerTuneArch, ProblemSize PROBLEM_SIZE, int CURRENT_BIT, int NUM_PASSES>
	struct IteratePasses<ProblemInstance, CompilerTuneArch, PROBLEM_SIZE, 0, CURRENT_BIT, NUM_PASSES>
	{
		/**
		 * DispatchPass pass
		 */
		template <int TUNE_ARCH>
		static cudaError_t DispatchPass(ProblemInstance &problem_instance)
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
			max_grid_size,
			debug);

		// The appropriate tuning arch-id from the arch-id targeted by the
		// active compiler pass.
		struct CompilerTuneArch
		{
			static const int VALUE 	=
				(__CUB_CUDA_ARCH__ >= 200) ?
					200 :
					(__CUB_CUDA_ARCH__ >= 130) ?
						130 :
						100;
		};

		typedef IteratePasses<CompilerTuneArch, ProblemInstance, PROBLEM_SIZE, BITS_REMAINING, CURRENT_BIT, 0> IteratePasses;

		if (cuda_props.kernel_ptx_version >= 200)
		{
			return IteratePasses::template DispatchPass<200>(problem_instance);
		}
		else if (cuda_props.kernel_ptx_version >= 130)
		{
			return IteratePasses::template DispatchPass<130>(problem_instance);
		}
		else
		{
			return IteratePasses::template DispatchPass<100>(problem_instance);
		}
	}
};





} // namespace radix_sort
} // namespace b40c

