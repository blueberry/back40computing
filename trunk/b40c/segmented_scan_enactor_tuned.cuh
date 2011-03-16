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
 * Tuned Scan Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/arch_dispatch.cuh>
#include <b40c/util/work_distribution.cuh>
#include <b40c/segmented_scan/segmented_scan_enactor.cuh>
#include <b40c/segmented_scan/problem_config_tuned.cuh>

namespace b40c {

/******************************************************************************
 * SegmentedScanEnactorTuned Declaration
 ******************************************************************************/

/**
 * Tuned segmented scan enactor class.
 */
class SegmentedScanEnactorTuned : public segmented_scan::SegmentedScanEnactor
{
public:

	//---------------------------------------------------------------------
	// Helper Structures (need to be public for cudafe)
	//---------------------------------------------------------------------

	template <typename T, typename Flag, typename SizeT> 		struct Storage;
	template <typename ProblemType> 							struct Detail;
	template <segmented_scan::ProbSizeGenre PROB_SIZE_GENRE> 	struct ConfigResolver;

protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Typedefs for base classes
	typedef segmented_scan::SegmentedScanEnactor BaseEnactorType;

	// Befriend our base types: they need to call back into a
	// protected methods (which are templated, and therefore can't be virtual)
	friend BaseEnactorType;


	//-----------------------------------------------------------------------------
	// Scan Operation
	//-----------------------------------------------------------------------------

    /**
	 * Performs a segmented scan pass
	 */
	template <typename TunedConfig>
	cudaError_t ScanPass(
		typename TunedConfig::T *d_dest,
		typename TunedConfig::T *d_src,
		typename TunedConfig::Downsweep::Flag *d_flag_src,
		util::CtaWorkDistribution<typename TunedConfig::Downsweep::SizeT> &work,
		int spine_elements);


public:

	/**
	 * Constructor.
	 */
	SegmentedScanEnactorTuned() : BaseEnactorType::SegmentedScanEnactor() {}


	/**
	 * Enacts a segmented scan on the specified device data using the specified
	 * granularity configuration  (ScanEnactor::Enact)
	 */
	using BaseEnactorType::Enact;


	/**
	 * Enacts a segmented scan operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param d_flag_src
	 * 		Pointer to array of "head flags" that demarcate independent scan segments
	 * @param num_elements
	 * 		Number of elements to segmented scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		typename Flag,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&),
		T Identity(),
		segmented_scan::ProbSizeGenre PROB_SIZE_GENRE>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		Flag *d_flag_src,
		size_t num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a segmented scan operation on the specified device data using
	 * a heuristic for selecting granularity configuration based upon
	 * problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param d_flag_src
	 * 		Pointer to array of "head flags" that demarcate independent scan segments
	 * @param num_elements
	 * 		Number of elements to segmented scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		typename Flag,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&),
		T Identity()>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		Flag *d_flag_src,
		size_t num_bytes,
		int max_grid_size = 0);
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename ProblemType>
struct SegmentedScanEnactorTuned::Detail
{
	typedef ProblemType Problem;
	
	SegmentedScanEnactorTuned *enactor;
	int max_grid_size;

	// Constructor
	Detail(SegmentedScanEnactorTuned *enactor, int max_grid_size = 0) :
		enactor(enactor), max_grid_size(max_grid_size) {}
};


/**
 * Type for encapsulating storage details regarding an invocation
 */
template <typename T, typename Flag, typename SizeT>
struct SegmentedScanEnactorTuned::Storage
{
	T *d_dest;
	T *d_src;
	Flag *d_flag_src;
	SizeT num_elements;

	// Constructor
	Storage(T *d_dest, T *d_src, Flag *d_flag_src, SizeT num_elements) :
		d_dest(d_dest), d_src(d_src), d_flag_src(d_flag_src), num_elements(num_elements) {}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
 */
template <segmented_scan::ProbSizeGenre PROB_SIZE_GENRE>
struct SegmentedScanEnactorTuned::ConfigResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		typedef typename DetailType::Problem ProblemType;

		// Obtain tuned granularity type
		typedef segmented_scan::TunedConfig<ProblemType, CUDA_ARCH, PROB_SIZE_GENRE> TunedConfig;

		// Invoke base class enact with type
		return detail.enactor->template EnactInternal<TunedConfig, SegmentedScanEnactorTuned>(
			storage.d_dest,
			storage.d_src,
			storage.d_flag_src,
			storage.num_elements,
			detail.max_grid_size);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
 */
template <>
struct SegmentedScanEnactorTuned::ConfigResolver <segmented_scan::UNKNOWN>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		typedef typename DetailType::Problem ProblemType;

		// Obtain large tuned granularity type
		typedef segmented_scan::TunedConfig<ProblemType, CUDA_ARCH, segmented_scan::LARGE> LargeConfig;

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargeConfig::Upsweep::TILE_ELEMENTS *
			LargeConfig::Upsweep::CTA_OCCUPANCY *
			detail.enactor->SmCount();

		if (storage.num_elements < saturating_load) {

			// Invoke base class enact with small-problem config type
			typedef segmented_scan::TunedConfig<ProblemType, CUDA_ARCH, segmented_scan::SMALL> SmallConfig;
			return detail.enactor->template EnactInternal<SmallConfig, SegmentedScanEnactorTuned>(
				storage.d_dest,
				storage.d_src,
				storage.d_flag_src,
				storage.num_elements,
				detail.max_grid_size);
		}

		// Invoke base class enact with type
		return detail.enactor->template EnactInternal<LargeConfig, SegmentedScanEnactorTuned>(
			storage.d_dest,
			storage.d_src,
			storage.d_flag_src,
			storage.num_elements,
			detail.max_grid_size);
	}
};


/******************************************************************************
 * SegmentedScanEnactorTuned Implementation
 ******************************************************************************/

/**
 * Performs a segmented scan pass
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 */
template <typename TunedConfig>
cudaError_t SegmentedScanEnactorTuned::ScanPass(
	typename TunedConfig::T *d_dest,
	typename TunedConfig::T *d_src,
	typename TunedConfig::Downsweep::Flag *d_flag_src,
	util::CtaWorkDistribution<typename TunedConfig::Downsweep::SizeT> &work,
	int spine_elements)
{
	using namespace segmented_scan;

	typedef typename TunedConfig::T T;
	typedef typename TunedConfig::Flag Flag;
	typedef typename TunedConfig::Upsweep Upsweep;
	typedef typename TunedConfig::Spine Spine;
	typedef typename TunedConfig::Downsweep Downsweep;
	typedef typename TunedConfig::Single Single;

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			TunedSpineScanSingleKernel<typename Single::ProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<1, Spine::THREADS, 0>>>(
				d_src, d_flag_src, d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor SpineScanKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option for dynamic smem allocation
			if (TunedConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, TunedUpsweepReductionKernel<typename Upsweep::ProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"ScanEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, TunedSpineScanKernel<typename Spine::ProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"ScanEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, TunedDownsweepScanKernel<typename Downsweep::ProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"ScanEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(
					upsweep_kernel_attrs.sharedSizeBytes,
					B40C_MAX(spine_kernel_attrs.sharedSizeBytes, downsweep_kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (TunedConfig::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep segmented scan into spine
			TunedUpsweepReductionKernel<typename Upsweep::ProblemType, TunedConfig::PROB_SIZE_GENRE>
				<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
					d_src, d_flag_src, (T*) partial_spine(), (Flag*) flag_spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor TunedUpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

			// Spine segmented scan
			TunedSpineScanKernel<typename Spine::ProblemType, TunedConfig::PROB_SIZE_GENRE>
				<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
					(T*) partial_spine(), (Flag*) flag_spine(), (T*) partial_spine(), spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor TunedSpineScanKernel failed ", __FILE__, __LINE__))) break;

			// Downsweep segmented scan into spine
			TunedDownsweepScanKernel<typename Downsweep::ProblemType, TunedConfig::PROB_SIZE_GENRE>
				<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
					d_src, d_flag_src, d_dest, (T*) partial_spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor TunedDownsweepScanKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}


/**
 * Enacts a segmented scan operation on the specified device data using the
 * enumerated tuned granularity configuration
 */
template <
	typename T,
	typename Flag,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity(),
	segmented_scan::ProbSizeGenre PROB_SIZE_GENRE>
cudaError_t SegmentedScanEnactorTuned::Enact(
	T *d_dest,
	T *d_src,
	Flag *d_flag_src,
	size_t num_elements,
	int max_grid_size)
{
	typedef size_t SizeT;
	typedef segmented_scan::ProblemType<T, Flag, SizeT, EXCLUSIVE, BinaryOp, Identity> Problem;
	typedef Detail<Problem> Detail;
	typedef Storage<T, Flag, SizeT> Storage;
	typedef ConfigResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, max_grid_size);
	Storage storage(d_dest, d_src, d_flag_src, num_elements);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(storage, detail, PtxVersion());
}


/**
 * Enacts a segmented scan operation on the specified device data using the
 * LARGE granularity configuration
 */
template <
	typename T,
	typename Flag,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity()>
cudaError_t SegmentedScanEnactorTuned::Enact(
	T *d_dest,
	T *d_src,
	Flag *d_flag_src,
	size_t num_elements,
	int max_grid_size)
{
	return Enact<T, EXCLUSIVE, BinaryOp, Identity, segmented_scan::UNKNOWN>(
		d_dest, d_src, d_flag_src, num_elements, max_grid_size);
}



} // namespace b40c

