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

#include <b40c/partition/problem_type.cuh>

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
    cudaError_t PreSort(Detail &detail);

	/**
     * Post-sorting logic.
     */
	template <typename Policy, typename Detail>
    cudaError_t PostSort(Detail &detail, int num_passes);

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
   			util::B40CPerror(cudaFree(d_selectors), "Enactor cudaFree d_selectors failed: ", __FILE__, __LINE__);
   		}
    }


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * If left NULL, the non-selected problem storage arrays will be allocated
	 * lazily upon the first sorting pass, and are the caller's responsibility
	 * for freeing. After a sorting operation has completed, the selector member will
	 * index the key (and value) pointers that contain the final sorted results.
	 * (E.g., an odd number of sorting passes may leave the results in d_keys[1] if
	 * the input started in d_keys[0].)
	 *
	 * @param problem_storage
	 * 		Instance of PingPongStorage type describing the details of the
	 * 		problem to sort.
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename PingPongStorage,
		typename SizeT>
	cudaError_t Sort(
		PingPongStorage &problem_storage,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a sorting operation on the specified device data.  Uses the
	 * specified problem size genre enumeration to select autotuning policy.
	 *
	 * (Using this entrypoint can save compile time by not compiling tuned
	 * kernels for each problem size genre.)
	 *
	 * If left NULL, the non-selected problem storage arrays will be allocated
	 * lazily upon the first sorting pass, and are the caller's responsibility
	 * for freeing. After a sorting operation has completed, the selector member will
	 * index the key (and value) pointers that contain the final sorted results.
	 * (E.g., an odd number of sorting passes may leave the results in d_keys[1] if
	 * the input started in d_keys[0].)
	 *
	 * @param problem_storage
	 * 		Instance of PingPongStorage type describing the details of the
	 * 		problem to sort.
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		ProbSizeGenre PROB_SIZE_GENRE,
		typename PingPongStorage,
		typename SizeT>
	cudaError_t Sort(
		PingPongStorage &problem_storage,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a sorting operation on the specified device data.  Uses the
	 * specified problem size genre enumeration to select autotuning policy.
	 *
	 * (Using this entrypoint can save compile time by not compiling tuned
	 * kernels for each problem size genre.)
	 *
	 * If left NULL, the non-selected problem storage arrays will be allocated
	 * lazily upon the first sorting pass, and are the caller's responsibility
	 * for freeing. After a sorting operation has completed, the selector member will
	 * index the key (and value) pointers that contain the final sorted results.
	 * (E.g., an odd number of sorting passes may leave the results in d_keys[1] if
	 * the input started in d_keys[0].)
	 *
	 * @param problem_storage
	 * 		Instance of PingPongStorage type describing the details of the
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
		ProbSizeGenre PROB_SIZE_GENRE,
		typename PingPongStorage,
		typename SizeT>
	cudaError_t Sort(
		PingPongStorage &problem_storage,
		SizeT num_elements,
		int max_grid_size = 0);


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
	 * 		Instance of PingPongStorage type describing the details of the
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
		typename Policy,
		typename PingPongStorage,
		typename SizeT>
	cudaError_t Sort(
		PingPongStorage &problem_storage,
		SizeT num_elements,
		int max_grid_size = 0);


    /**
	 * Performs a radix sorting operation
	 */
	template <typename Policy, typename Detail>
	cudaError_t EnactSort(Detail &detail);

	/**
	 * Performs a radix sorting pass
	 */
	template <typename Policy, typename PassPolicy, typename Detail>
	cudaError_t EnactPass(Detail &detail);
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <
	int _START_BIT,
	int _NUM_BITS,
	typename PingPongStorage,
	typename SizeT>
struct Detail
{
	static const int START_BIT 		= _START_BIT;
	static const int NUM_BITS 		= _NUM_BITS;

	// Key conversion trait type
	typedef KeyTraits<typename PingPongStorage::KeyType> KeyTraits;

	// Problem type is on unsigned keys (converted key type)
	typedef partition::ProblemType<
		typename KeyTraits::ConvertedKeyType,
		typename PingPongStorage::ValueType,
		SizeT> ProblemType;

	// Problem data
	Enactor 			*enactor;
	PingPongStorage 	&problem_storage;
	SizeT				num_elements;
	int			 		max_grid_size;

	// Operational details
	util::CtaWorkDistribution<SizeT> 	work;
	SizeT 								spine_elements;

	// Constructor
	Detail(
		Enactor *enactor,
		PingPongStorage &problem_storage,
		SizeT num_elements,
		int max_grid_size = 0) :
			enactor(enactor),
			num_elements(num_elements),
			problem_storage(problem_storage),
			max_grid_size(max_grid_size)
	{}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
 */
template <ProbSizeGenre PROB_SIZE_GENRE>
struct PolicyResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		typedef typename Detail::ProblemType ProblemType;

		// Obtain tuned granularity type
		typedef AutotunedPolicy<ProblemType, CUDA_ARCH, PROB_SIZE_GENRE> AutotunedPolicy;

		// Invoke base class enact with type
		return detail.enactor->template EnactSort<AutotunedPolicy>(detail);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
 */
template <>
struct PolicyResolver <UNKNOWN_SIZE>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		typedef typename Detail::ProblemType ProblemType;

		// Obtain large tuned granularity type
		typedef AutotunedPolicy<ProblemType, CUDA_ARCH, LARGE_SIZE> LargePolicy;

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargePolicy::Upsweep::TILE_ELEMENTS *
			LargePolicy::Upsweep::CTA_OCCUPANCY *
			detail.enactor->SmCount();

		if (detail.num_elements < saturating_load) {

			// Invoke base class enact with small-problem config type
			typedef AutotunedPolicy<ProblemType, CUDA_ARCH, SMALL_SIZE> SmallPolicy;
			return detail.enactor->template EnactSort<SmallPolicy>(detail);
		}

		// Invoke base class enact with type
		return detail.enactor->template EnactSort<LargePolicy>(detail);
	}
};


/**
 * Iteration structure for unrolling sorting passes
 */
template <typename Policy>
struct PassIteration
{
	/**
	 * Middle sorting passes (i.e., neither first, nor last pass).  Does not apply
	 * any pre/post bit-twiddling functors.
	 */
	template <
		int CURRENT_PASS,
		int LAST_PASS,
		int CURRENT_BIT,
		int RADIX_BITS = Policy::Upsweep::LOG_BINS>
	struct Iterate
	{
		template <typename Detail>
		static cudaError_t Invoke(Enactor *enactor, Detail &detail)
		{
			typedef PassPolicy<CURRENT_PASS, CURRENT_BIT, NopKeyConversion, NopKeyConversion> PassPolicy;

			cudaError_t retval = enactor->template EnactPass<Policy, PassPolicy>(detail);
			if (retval) return retval;

			return Iterate<
				CURRENT_PASS + 1,
				LAST_PASS,
				CURRENT_BIT + RADIX_BITS,
				RADIX_BITS>::Invoke(enactor, detail);
		}
	};

	/**
	 * First sorting pass (unless there's only one pass).  Applies the
	 * appropriate pre-process bit-twiddling functor.
	 */
	template <
		int LAST_PASS,
		int CURRENT_BIT,
		int RADIX_BITS>
	struct Iterate <0, LAST_PASS, CURRENT_BIT, RADIX_BITS>
	{
		template <typename Detail>
		static cudaError_t Invoke(Enactor *enactor, Detail &detail)
		{
			typedef PassPolicy<0, CURRENT_BIT, typename Detail::KeyTraits, NopKeyConversion> PassPolicy;

			cudaError_t retval = enactor->template EnactPass<Policy, PassPolicy>(detail);
			if (retval) return retval;

			return Iterate<
				1,
				LAST_PASS,
				CURRENT_BIT + RADIX_BITS,
				RADIX_BITS>::Invoke(enactor, detail);
		}
	};

	/**
	 * Last sorting pass (unless there's only one pass).  Applies the
	 * appropriate post-process bit-twiddling functor.
	 */
	template <
		int LAST_PASS,
		int CURRENT_BIT,
		int RADIX_BITS>
	struct Iterate <LAST_PASS, LAST_PASS, CURRENT_BIT, RADIX_BITS>
	{
		template <typename Detail>
		static cudaError_t Invoke(Enactor *enactor, Detail &detail)
		{
			typedef PassPolicy<LAST_PASS, CURRENT_BIT, NopKeyConversion, typename Detail::KeyTraits> PassPolicy;

			return enactor->template EnactPass<Policy, PassPolicy>(detail);
		}
	};

	/**
	 * Singular sorting pass (when there's only one pass).  Applies both
	 * pre- and post-process bit-twiddling functors.
	 */
	template <
		int CURRENT_BIT,
		int RADIX_BITS>
	struct Iterate <0, 0, CURRENT_BIT, RADIX_BITS>
	{
		template <typename Detail>
		static cudaError_t Invoke(Enactor *enactor, Detail &detail)
		{
			typedef PassPolicy<0, CURRENT_BIT, typename Detail::KeyTraits, typename Detail::KeyTraits> PassPolicy;

			return enactor->template EnactPass<Policy, PassPolicy>(detail);
		}
	};
};




/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/

/**
 * Performs a radix sorting pass
 */
template <typename Policy, typename PassPolicy, typename Detail>
cudaError_t Enactor::EnactPass(Detail &detail)
{

	// Policy
	typedef typename Policy::Upsweep 						Upsweep;
	typedef typename Policy::Spine 							Spine;
	typedef typename Policy::Downsweep 						Downsweep;

	// Data types
	typedef typename Policy::KeyType 						KeyType;
	typedef typename Policy::ValueType 						ValueType;
	typedef typename Policy::SizeT 							SizeT;
	typedef typename Detail::KeyTraits::ConvertedKeyType 	ConvertedKeyType;

	cudaError_t retval = cudaSuccess;

	// Kernel pointers
	typename Policy::UpsweepKernelPtr UpsweepKernel = Policy::template UpsweepKernel<PassPolicy>();
	typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();
	typename Policy::DownsweepKernelPtr DownsweepKernel = Policy::template DownsweepKernel<PassPolicy>();


	do {

		int dynamic_smem[3] = 	{0, 0, 0};
		int grid_size[3] = 		{detail.work.grid_size, 1, detail.work.grid_size};

		// Tuning option for dynamic smem allocation
		if (Policy::UNIFORM_SMEM_ALLOCATION) {

			// We need to compute dynamic smem allocations to ensure all three
			// kernels end up allocating the same amount of smem per CTA

			// Get kernel attributes
			cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepKernel),
				"Enactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineKernel),
				"Enactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, DownsweepKernel),
				"Enactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

			int max_static_smem = B40C_MAX(
				upsweep_kernel_attrs.sharedSizeBytes,
				B40C_MAX(spine_kernel_attrs.sharedSizeBytes, downsweep_kernel_attrs.sharedSizeBytes));

			dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
			dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
			dynamic_smem[2] = max_static_smem - downsweep_kernel_attrs.sharedSizeBytes;
		}

		// Tuning option for spine-sort kernel grid size
		if (Policy::UNIFORM_GRID_SIZE) {
			grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
		}

		// Upsweep reduction into spine
		UpsweepKernel<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
			d_selectors,
			(SizeT*) spine(),
			(ConvertedKeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector],
			(ConvertedKeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector ^ 1],
			detail.work);

		if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__))) break;

		// Spine scan
		SpineKernel<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
			(SizeT*) spine(), (SizeT*) spine(), detail.spine_elements);

		if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;

		// Downsweep scan from spine
		DownsweepKernel<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
			d_selectors,
			(SizeT *) spine(),
			(ConvertedKeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector],
			(ConvertedKeyType *) detail.problem_storage.d_keys[detail.problem_storage.selector ^ 1],
			detail.problem_storage.d_values[detail.problem_storage.selector],
			detail.problem_storage.d_values[detail.problem_storage.selector ^ 1],
			detail.work);

		if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__))) break;

	} while (0);

	return retval;
}


/**
 * Pre-sorting logic.
 */
template <typename Policy, typename Detail>
cudaError_t Enactor::PreSort(Detail &detail)
{
	typedef typename Policy::KeyType 		KeyType;
	typedef typename Policy::ValueType 		ValueType;
	typedef typename Policy::SizeT 			SizeT;

	cudaError_t retval = cudaSuccess;
	do {
		// Setup d_selectors if necessary
		if (d_selectors == NULL) {
			if (retval = util::B40CPerror(cudaMalloc((void**) &d_selectors, 2 * sizeof(int)),
				"LsbSortEnactor cudaMalloc d_selectors failed", __FILE__, __LINE__)) break;
		}

		// Setup pong-storage if necessary
		if (detail.problem_storage.d_keys[0] == NULL) {
			if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_keys[0], detail.num_elements * sizeof(KeyType)),
				"LsbSortEnactor cudaMalloc detail.problem_storage.d_keys[0] failed", __FILE__, __LINE__)) break;
		}
		if (detail.problem_storage.d_keys[1] == NULL) {
			if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_keys[1], detail.num_elements * sizeof(KeyType)),
				"LsbSortEnactor cudaMalloc detail.problem_storage.d_keys[1] failed", __FILE__, __LINE__)) break;
		}
		if (!util::Equals<ValueType, util::NullType>::VALUE) {
			if (detail.problem_storage.d_values[0] == NULL) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_values[0], detail.num_elements * sizeof(ValueType)),
					"LsbSortEnactor cudaMalloc detail.problem_storage.d_values[0] failed", __FILE__, __LINE__)) break;
			}
			if (detail.problem_storage.d_values[1] == NULL) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &detail.problem_storage.d_values[1], detail.num_elements * sizeof(ValueType)),
					"LsbSortEnactor cudaMalloc detail.problem_storage.d_values[1] failed", __FILE__, __LINE__)) break;
			}
		}

		// Make sure our spine is big enough
		if (retval = spine.Setup<SizeT>(detail.spine_elements)) break;

	} while (0);

	return retval;
}


/**
 * Post-sorting logic.
 */
template <typename Policy, typename Detail>
cudaError_t Enactor::PostSort(Detail &detail, int num_passes)
{
	cudaError_t retval = cudaSuccess;

	do {
		if (!Policy::Upsweep::EARLY_EXIT) {

			// We moved data between storage buffers at every pass
			detail.problem_storage.selector = (detail.problem_storage.selector + num_passes) & 0x1;

		} else {

			// Save old selector
			int old_selector = detail.problem_storage.selector;

			// Copy out the selector from the last pass
			if (retval = util::B40CPerror(cudaMemcpy(&detail.problem_storage.selector, &d_selectors[num_passes & 0x1], sizeof(int), cudaMemcpyDeviceToHost),
				"LsbSortEnactor cudaMemcpy d_selector failed", __FILE__, __LINE__)) break;

			// Correct new selector if the original indicated that we started off from the alternate
			detail.problem_storage.selector ^= old_selector;
		}

	} while (0);

	return retval;
}

/**
 * Enacts a sort on the specified device data.
 */
template <typename Policy, typename Detail>
cudaError_t Enactor::EnactSort(Detail &detail)
{
	// Policy
	typedef typename Policy::Upsweep 	Upsweep;
	typedef typename Policy::Spine 		Spine;
	typedef typename Policy::Downsweep 	Downsweep;

	// Data types
	typedef typename Policy::KeyType	KeyType;
	typedef typename Policy::ValueType	ValueType;
	typedef typename Policy::SizeT 		SizeT;

	const int NUM_PASSES 				= (Detail::NUM_BITS + Downsweep::LOG_BINS - 1) / Downsweep::LOG_BINS;
	const int MIN_OCCUPANCY 			= B40C_MIN((int) Upsweep::MAX_CTA_OCCUPANCY, (int) Downsweep::MAX_CTA_OCCUPANCY);
	util::SuppressUnusedConstantWarning(MIN_OCCUPANCY);

	// Compute sweep grid size
	int grid_size = (Policy::OVERSUBSCRIBED_GRID_SIZE) ?
		OversubscribedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(detail.num_elements, detail.max_grid_size) :
		OccupiedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(detail.num_elements, detail.max_grid_size);

	// Compute spine elements (round up to nearest spine tile_elements)
	detail.spine_elements = grid_size << Downsweep::LOG_BINS;
	detail.spine_elements = ((detail.spine_elements + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) * Spine::TILE_ELEMENTS;

	// Obtain a CTA work distribution for copying items of type T
	detail.work.template Init<Downsweep::LOG_SCHEDULE_GRANULARITY>(
		detail.num_elements, grid_size);

	if (DEBUG) {
		printf("\n\n");
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version,
			cuda_props.kernel_ptx_version);
		printf("Sorting: \t[radix_bits: %d, start_bit: %d, num_bits: %d, num_passes: %d]\n",
			Downsweep::LOG_BINS,
			Detail::START_BIT,
			Detail::NUM_BITS,
			NUM_PASSES);
		printf("Upsweep: \t[grid_size: %d, threads %d, tile_elements: %d]\n",
			detail.work.grid_size,
			Upsweep::THREADS,
			Upsweep::TILE_ELEMENTS);
		printf("Spine: \t\t[threads: %d, spine_elements: %d, tile_elements: %d]\n",
			Spine::THREADS,
			detail.spine_elements,
			Spine::TILE_ELEMENTS);
		printf("Downsweep: \t[grid_size: %d, threads %d, tile_elements: %d]\n",
			detail.work.grid_size,
			Downsweep::THREADS,
			Downsweep::TILE_ELEMENTS);
		printf("Work: \t\t[key bytes: %lu, value bytes: %lu, SizeT %lu bytes, num_elements: %lu, schedule_granularity: %d, total_grains: %lu, grains_per_cta: %lu, extra_grains: %lu]\n",
			(unsigned long) sizeof(KeyType),
			(unsigned long) sizeof(ValueType),
			(unsigned long) sizeof(SizeT),
			(unsigned long) detail.num_elements,
			Downsweep::SCHEDULE_GRANULARITY,
			(unsigned long) detail.work.total_grains,
			(unsigned long) detail.work.grains_per_cta,
			(unsigned long) detail.work.extra_grains);
		fflush(stdout);
	}

	cudaError_t retval = cudaSuccess;

	do {

		// Perform any preparation prior to sorting
		if (retval = PreSort<Policy>(detail)) break;

		// Perform sorting passes
		if (retval = PassIteration<Policy>::template Iterate<
				0,
				NUM_PASSES - 1,
				Detail::START_BIT,
				Downsweep::LOG_BINS>::Invoke(this, detail)) break;

		// Perform any cleanup after sorting
		if (retval = PostSort<Policy>(detail, NUM_PASSES)) break;

	} while (0);

	return retval;
}


/**
 * Enacts a sort on the specified device data.
 */
template <
	int START_BIT,
	int NUM_BITS,
	typename Policy,
	typename PingPongStorage,
	typename SizeT>
cudaError_t Enactor::Sort(
	PingPongStorage &problem_storage,
	SizeT num_elements,
	int max_grid_size)
{
	typedef Detail<START_BIT, NUM_BITS, PingPongStorage, SizeT> Detail;

	Detail detail(this, problem_storage, num_elements, max_grid_size);

	return EnactSort<Policy, Detail>(detail);
}

/**
 * Enacts a sort operation on the specified device data.
 */
template <
	int START_BIT,
	int NUM_BITS,
	ProbSizeGenre PROB_SIZE_GENRE,
	typename PingPongStorage,
	typename SizeT>
cudaError_t Enactor::Sort(
	PingPongStorage &problem_storage,
	SizeT num_elements,
	int max_grid_size)
{
	typedef Detail<START_BIT, NUM_BITS, PingPongStorage, SizeT> Detail;
	typedef PolicyResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, problem_storage, num_elements, max_grid_size);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(
		detail, PtxVersion());
}


/**
 * Enacts a sort operation on the specified device data.
 */
template <
	ProbSizeGenre PROB_SIZE_GENRE,
	typename PingPongStorage,
	typename SizeT>
cudaError_t Enactor::Sort(
	PingPongStorage &problem_storage,
	SizeT num_elements,
	int max_grid_size)
{
	return Sort<0, sizeof(typename PingPongStorage::KeyType) * 8, PROB_SIZE_GENRE>(
		problem_storage, num_elements, max_grid_size);
}


/**
 * Enacts a sort operation on the specified device data.
 */
template <
	typename PingPongStorage,
	typename SizeT>
cudaError_t Enactor::Sort(
	PingPongStorage &problem_storage,
	SizeT num_elements,
	int max_grid_size)
{
	return Sort<UNKNOWN_SIZE>(
		problem_storage, num_elements, max_grid_size);
}



} // namespace radix_sort
} // namespace b40c

