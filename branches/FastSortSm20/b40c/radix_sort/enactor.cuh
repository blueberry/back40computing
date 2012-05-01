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

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/arch_dispatch.cuh>
#include <b40c/util/ping_pong_storage.cuh>
#include <b40c/util/numeric_traits.cuh>

#include <b40c/radix_sort/problem_type.cuh>
#include <b40c/radix_sort/policy.cuh>
#include <b40c/radix_sort/pass_policy.cuh>
#include <b40c/radix_sort/autotuned_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel.cuh>
#include <b40c/radix_sort/upsweep/kernel.cuh>

#include <b40c/scan/spine/kernel.cuh>

namespace b40c {
namespace radix_sort {


/**
 * Radix sorting enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine spine;

	// Pair of "selector" device integers.  The first selects the incoming device
	// vector for even passes, the second selects the odd.
	int *d_selectors;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

    /**
     * Pre-sorting logic.
     */
	template <typename Policy, typename Detail>
    cudaError_t PreSort(Detail &detail)
	{
		typedef typename Policy::KeyType 		KeyType;
		typedef typename Policy::ValueType 		ValueType;
		typedef typename Policy::SizeT 			SizeT;

		cudaError_t retval = cudaSuccess;
		do {
			// Setup d_selectors if necessary
			if (d_selectors == NULL) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_selectors, 2 * sizeof(int)),
					"LsbSortEnactor cudaMalloc d_selectors failed", __FILE__, __LINE__, ENACTOR_DEBUG)) break;
			}

			// Setup pong-storage if necessary
			if (detail.problem_storage.d_keys[0] == NULL) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_keys[0], detail.num_elements * sizeof(KeyType)),
					"LsbSortEnactor cudaMalloc detail.problem_storage.d_keys[0] failed", __FILE__, __LINE__, ENACTOR_DEBUG)) break;
			}
			if (detail.problem_storage.d_keys[1] == NULL) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_keys[1], detail.num_elements * sizeof(KeyType)),
					"LsbSortEnactor cudaMalloc detail.problem_storage.d_keys[1] failed", __FILE__, __LINE__, ENACTOR_DEBUG)) break;
			}
			if (!util::Equals<ValueType, util::NullType>::VALUE) {
				if (detail.problem_storage.d_values[0] == NULL) {
					if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_values[0], detail.num_elements * sizeof(ValueType)),
						"LsbSortEnactor cudaMalloc detail.problem_storage.d_values[0] failed", __FILE__, __LINE__, ENACTOR_DEBUG)) break;
				}
				if (detail.problem_storage.d_values[1] == NULL) {
					if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_values[1], detail.num_elements * sizeof(ValueType)),
						"LsbSortEnactor cudaMalloc detail.problem_storage.d_values[1] failed", __FILE__, __LINE__, ENACTOR_DEBUG)) break;
				}
			}

		} while (0);

		return retval;
	}


	/**
     * Post-sorting logic.
     */
	template <typename Policy, typename Detail>
    cudaError_t PostSort(Detail &detail)
	{
		cudaError_t retval = cudaSuccess;

		do {

	//		if (!Policy::Upsweep::EARLY_EXIT) {

				// We moved data between storage buffers at every pass
				detail.problem_storage.selector = (detail.problem_storage.selector + detail.num_passes) & 0x1;
	/*
			} else {

				// Save old selector
				int old_selector = detail.problem_storage.selector;

				// Copy out the selector from the last pass
				if (retval = util::B40CPerror(cudaMemcpy(&detail.problem_storage.selector, &d_selectors[num_passes & 0x1], sizeof(int), cudaMemcpyDeviceToHost),
					"LsbSortEnactor cudaMemcpy d_selector failed", __FILE__, __LINE__, ENACTOR_DEBUG)) break;

				// Correct new selector if the original indicated that we started off from the alternate
				detail.problem_storage.selector ^= old_selector;
			}
	*/
		} while (0);

		return retval;
	}


	/**
	 * Bind value textures (specialized for primitive types)
	 */
	template <
		typename BitPolicy,
		typename ValueType,
		int REPRESENTATION = util::NumericTraits<ValueType>::REPRESENTATION>
	struct BindValueTextures
	{
		template <typename Detail>
		static cudaError_t Bind(Detail &detail)
		{
			typedef typename util::VecType<ValueType, BitPolicy::Downsweep::PACK_SIZE>::Type ValueVectorType;
			cudaChannelFormatDesc values_tex_desc = cudaCreateChannelDesc<ValueVectorType>();

			cudaError_t retval = cudaSuccess;
			do {
				if (retval = util::B40CPerror(cudaBindTexture(
						0,
						partition::downsweep::ValuesTex<ValueVectorType>::ref0,
						detail.problem_storage.d_values[detail.problem_storage.selector],
						values_tex_desc,
						detail.num_elements * sizeof(ValueVectorType)),
					"EnactorTwoPhase cudaBindTexture ValuesTex failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaBindTexture(
						0,
						partition::downsweep::ValuesTex<ValueVectorType>::ref1,
						detail.problem_storage.d_values[detail.problem_storage.selector ^ 1],
						values_tex_desc,
						detail.num_elements * sizeof(ValueVectorType)),
					"EnactorTwoPhase cudaBindTexture ValuesTex failed", __FILE__, __LINE__)) break;

			} while(0);

			return retval;
		}

	};


	/**
	 * Bind value textures (specialized for non-primitive types)
	 */
	template <
		typename BitPolicy,
		typename ValueType>
	struct BindValueTextures<BitPolicy, ValueType, util::NOT_A_NUMBER>
	{
		template <typename Detail>
		static cudaError_t Bind(Detail &detail)
		{
			// do nothing
			return cudaSuccess;
		}
	};


	/**
	 * Performs a radix sorting pass
	 */
	template <typename BitPolicy, typename PassPolicy, typename Detail>
	cudaError_t EnactPass(Detail &detail)
	{
		// Tuning policy
		typedef typename BitPolicy::Upsweep 					Upsweep;
		typedef typename BitPolicy::Spine 						Spine;
		typedef typename BitPolicy::Downsweep 					Downsweep;

		// Data types
		typedef typename BitPolicy::KeyType 					KeyType;	// Converted key type
		typedef typename BitPolicy::ValueType 					ValueType;
		typedef typename BitPolicy::SizeT 						SizeT;

		cudaError_t retval = cudaSuccess;

		do {
			if (ENACTOR_DEBUG) {
				printf("Pass %d, Bit %d, Radix bits %d:\n",
					PassPolicy::CURRENT_PASS,
					PassPolicy::CURRENT_BIT,
					BitPolicy::RADIX_BITS);
			}

			// Kernel pointers
			typename BitPolicy::UpsweepKernelPtr 		UpsweepKernel 		= BitPolicy::template UpsweepKernel<PassPolicy>();
			typename BitPolicy::SpineKernelPtr 			SpineKernel 		= BitPolicy::template SpineKernel<PassPolicy>();
			typename BitPolicy::DownsweepKernelPtr		DownsweepKernel 	= BitPolicy::template DownsweepKernel<PassPolicy>();

			// Max CTA occupancy for the actual target device
			int upsweep_cta_occupancy, downsweep_cta_occupancy;
			if (retval = MaxCtaOccupancy(
				upsweep_cta_occupancy,
				UpsweepKernel,
				Upsweep::THREADS)) break;
			if (retval = MaxCtaOccupancy(
				downsweep_cta_occupancy,
				DownsweepKernel,
				Downsweep::THREADS)) break;

			if (ENACTOR_DEBUG) printf("Upsweep occupancy %d, downsweep occupancy %d\n", upsweep_cta_occupancy, downsweep_cta_occupancy);

			int sweep_grid_size = GridSize(
				true, 										// oversubscribed
				Upsweep::SCHEDULE_GRANULARITY,
				CUB_MIN(upsweep_cta_occupancy, downsweep_cta_occupancy),
				detail.num_elements,
				detail.max_grid_size);

			// Compute spine elements: BIN elements per CTA, rounded
			// up to nearest spine tile size
			SizeT spine_elements = sweep_grid_size << Downsweep::LOG_BINS;
			spine_elements = ((spine_elements + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) * Spine::TILE_ELEMENTS;

			// Make sure our spine is big enough
			if (retval = spine.Setup<SizeT>(spine_elements)) break;

			// Bind spine textures
			cudaChannelFormatDesc spine_tex_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					partition::spine::SpineTex<SizeT>::ref,
					(SizeT *) spine(),
					spine_tex_desc,
					spine_elements * sizeof(SizeT)),
				"EnactorTwoPhase cudaBindTexture SpineTex failed", __FILE__, __LINE__)) break;

			// Bind key textures
			typedef typename util::VecType<KeyType, BitPolicy::Downsweep::PACK_SIZE>::Type KeyVectorType;
			cudaChannelFormatDesc keys_tex_desc = cudaCreateChannelDesc<KeyVectorType>();

			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					partition::downsweep::KeysTex<KeyVectorType>::ref0,
					detail.problem_storage.d_keys[detail.problem_storage.selector],
					keys_tex_desc,
					detail.num_elements * sizeof(KeyVectorType)),
				"EnactorTwoPhase cudaBindTexture KeysTex failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					partition::downsweep::KeysTex<KeyVectorType>::ref1,
					detail.problem_storage.d_keys[detail.problem_storage.selector ^ 1],
					keys_tex_desc,
					detail.num_elements * sizeof(KeyVectorType)),
				"EnactorTwoPhase cudaBindTexture KeysTex failed", __FILE__, __LINE__)) break;

			// Bind value textures
			if (retval = BindValueTextures<BitPolicy, ValueType>::Bind(detail)) break;

			// Obtain a CTA work distribution
			util::CtaWorkDistribution<SizeT> work;
			work.template Init<Downsweep::LOG_SCHEDULE_GRANULARITY>(detail.num_elements, sweep_grid_size);

			if (ENACTOR_DEBUG) {
				PrintPassInfo<Upsweep, Spine, Downsweep>(work, spine_elements);
				printf("\n");
				fflush(stdout);
			}

			// Operational details
			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{sweep_grid_size, 1, sweep_grid_size};

			// Tuning option: make sure that all kernels launch the same number of CTAs)
			if (BitPolicy::UNIFORM_GRID_SIZE) grid_size[1] = grid_size[0];

			// Dynamic smem padding
			if (BitPolicy::UNIFORM_SMEM_ALLOCATION) {
				// Make sure all kernels have the same overall smem allocation
				if (retval = PadUniformSmem(dynamic_smem, UpsweepKernel, SpineKernel, DownsweepKernel)) break;
			} else {
				// Compute smem padding for upsweep to make upsweep occupancy a multiple of downsweep occupancy
				KernelDetails upsweep_details(UpsweepKernel, grid_size[0], this->cuda_props);
				dynamic_smem[0] = upsweep_details.SmemPadding(downsweep_cta_occupancy);
			}

			// Upsweep reduction into spine
			UpsweepKernel<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
				d_selectors,
				(SizeT*) spine(),
				(KeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector],
				(KeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector ^ 1],
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Spine scan
			SpineKernel<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(SizeT*) spine(), (SizeT*) spine(), spine_elements);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Set shared mem bank mode
			enum cudaSharedMemConfig old_config;
			cudaDeviceGetSharedMemConfig(&old_config);
			cudaDeviceSetSharedMemConfig((sizeof(typename Downsweep::RakingPartial) > 4) ?
				cudaSharedMemBankSizeEightByte :		// 64-bit bank mode
				cudaSharedMemBankSizeFourByte);			// 32-bit bank mode

			// Downsweep scan from spine
			DownsweepKernel<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
				d_selectors,
				(SizeT *) spine(),
				(KeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector],
				(KeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector ^ 1],
				detail.problem_storage.d_values[detail.problem_storage.selector],
				detail.problem_storage.d_values[detail.problem_storage.selector ^ 1],
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Restore smem bank mode
			cudaDeviceSetSharedMemConfig(old_config);

		} while (0);

		return retval;
	}


	//-----------------------------------------------------------------------------
	// Helper structures
	//-----------------------------------------------------------------------------

	/**
	 * Type for encapsulating operational details regarding an invocation
	 */
	template <typename _ProblemType>
	struct Detail
	{
		// Problem type
		typedef _ProblemType 														ProblemType;
		typedef typename ProblemType::OriginalKeyType								StorageKeyType;
		typedef typename ProblemType::ValueType										StorageValueType;
		typedef typename ProblemType::SizeT											SizeT;
		typedef typename util::PingPongStorage<StorageKeyType, StorageValueType> 	PingPongStorage;

		// Problem data
		PingPongStorage		&problem_storage;
		SizeT				num_elements;
		int			 		max_grid_size;
		int					num_passes;
		Enactor				*enactor;

		// Constructor
		Detail(
			Enactor *enactor,
			PingPongStorage &problem_storage,
			SizeT num_elements,
			int max_grid_size = 0) :
				enactor(enactor),
				num_elements(num_elements),
				problem_storage(problem_storage),
				max_grid_size(max_grid_size),
				num_passes(0)
		{}

	};


	/**
	 * Middle sorting passes (i.e., neither first, nor last pass).  Does not apply
	 * any pre/post bit-twiddling functors.
	 */
	template <
		typename Policy,
		int FIRST_BIT,
		int CURRENT_BIT,
		int LAST_BIT,
		int CURRENT_PASS>
	struct PassIteration
	{
		typedef PassPolicy<
			CURRENT_PASS,
			CURRENT_BIT,
			NopKeyConversion,
			NopKeyConversion> PassPolicy;

		enum {
			BITS_LEFT 				= LAST_BIT - CURRENT_BIT,
			BIT_SELECT 				= ((BITS_LEFT > 12) || (BITS_LEFT % 5 == 0)) ?
											5 :
											((BITS_LEFT > 3) || (BITS_LEFT % 4 == 0)) ?
												4 :
												BITS_LEFT
		};

		typedef typename Policy::template BitPolicy<BIT_SELECT>::Policy BitPolicy;

		enum {
			BITS_RUN = CUB_MIN(BITS_LEFT, BitPolicy::RADIX_BITS)
		};

		template <typename Detail>
		static cudaError_t Invoke(Detail &detail)
		{
			cudaError_t retval = detail.enactor->template EnactPass<BitPolicy, PassPolicy>(detail);
			if (retval) return retval;

			detail.num_passes++;

			return PassIteration<
				Policy,
				FIRST_BIT,
				CURRENT_BIT + BITS_RUN,
				LAST_BIT,
				CURRENT_PASS + 1>::Invoke(detail);
		}
	};


	/**
	 * Done
	 */
	template <
		typename Policy,
		int FIRST_BIT,
		int CURRENT_BIT,
		int CURRENT_PASS>
	struct PassIteration <Policy, FIRST_BIT, CURRENT_BIT, CURRENT_BIT, CURRENT_PASS>
	{
		template <typename Detail>
		static cudaError_t Invoke(Detail &detail)
		{
			return cudaSuccess;
		}
	};


public:

	/**
	 * Constructor
	 */
	Enactor() : d_selectors(NULL) {}


	/**
     * Destructor
     */
    virtual ~Enactor()
    {
   		if (d_selectors) {
   			util::B40CPerror(cudaFree(d_selectors), "Enactor cudaFree d_selectors failed: ", __FILE__, __LINE__, ENACTOR_DEBUG);
   		}
    }

	/**
	 * Enacts a scan on the specified device data.  Uses the specified
	 * kernel configuration policy.  (Useful for auto-tuning.)
	 *
	 * If left NULL, the non-selected problem storage arrays will be allocated
	 * lazily upon the first sorting pass, and are the caller's responsibility
	 * for freeing. After a sorting operation has completed, the selector member will
	 * index the key (and value) pointers that contain the final sorted results.
	 * (E.g., an odd number of sorting passes may leave the results in d_keys[1] if
	 * the input started in d_keys[0].)
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::PingPongStorage type describing the details of the
	 * 		problem to sort.
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		int START_BIT,
		int NUM_BITS,
		typename Policy>
	cudaError_t Sort(
		util::PingPongStorage<
			typename Policy::OriginalKeyType,
			typename Policy::ValueType> &problem_storage,
		typename Policy::SizeT num_elements,
		int max_grid_size = 0)
	{
		Detail<Policy> detail(
			this,
			problem_storage,
			num_elements,
			max_grid_size);

		cudaError_t retval = cudaSuccess;
		do {

			if (ENACTOR_DEBUG) {
				printf("\n\n");
				printf("Sorting: \t[start_bit: %d, num_bits: %d]\n",
					START_BIT,
					NUM_BITS);
				fflush(stdout);
			}

			// Perform any preparation prior to sorting
			if (retval = PreSort<Policy>(detail)) break;

			// Perform sorting passes
			if (retval = PassIteration<
				Policy,
				START_BIT,
				START_BIT,
				START_BIT + NUM_BITS,
				0>::Invoke(detail)) break;

			// Perform any cleanup after sorting
			if (retval = PostSort<Policy>(detail)) break;

		} while (0);

		return retval;
	}

};





} // namespace radix_sort
} // namespace b40c

