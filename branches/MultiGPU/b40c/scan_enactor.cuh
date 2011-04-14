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
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/scan/enactor.cuh>
#include <b40c/scan/problem_config_tuned.cuh>
#include <b40c/scan/problem_type.cuh>

namespace b40c {

/******************************************************************************
 * ScanEnactor Declaration
 ******************************************************************************/

/**
 * Tuned scan enactor class.
 */
class ScanEnactor : public scan::Enactor
{
public:

	//---------------------------------------------------------------------
	// Helper Structures (need to be public for cudafe)
	//---------------------------------------------------------------------

	template <typename T, typename SizeT> 					struct Storage;
	template <typename ProblemType> 						struct Detail;
	template <scan::ProbSizeGenre PROB_SIZE_GENRE> 			struct ConfigResolver;

protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Typedefs for base classes
	typedef scan::Enactor BaseEnactorType;

	// Befriend our base types: they need to call back into a
	// protected methods (which are templated, and therefore can't be virtual)
	friend BaseEnactorType;


	//-----------------------------------------------------------------------------
	// Scan Operation
	//-----------------------------------------------------------------------------

    /**
	 * Performs a scan pass
	 */
	template <typename TunedConfig>
	cudaError_t EnactPass(
		typename TunedConfig::T *d_dest,
		typename TunedConfig::T *d_src,
		util::CtaWorkDistribution<typename TunedConfig::SizeT> &work,
		int spine_elements);


public:

	/**
	 * Constructor.
	 */
	ScanEnactor() : BaseEnactorType::Enactor() {}


	/**
	 * Enacts a scan on the specified device data using the specified
	 * granularity configuration  (Enactor::Enact)
	 */
	using BaseEnactorType::Enact;


	/**
	 * Enacts a scan operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param num_elements
	 * 		Number of elements to scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&),
		T Identity(),
		scan::ProbSizeGenre PROB_SIZE_GENRE,
		typename SizeT>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a scan operation on the specified device data using
	 * a heuristic for selecting granularity configuration based upon
	 * problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param num_elements
	 * 		Number of elements to scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&),
		T Identity(),
		typename SizeT>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0);

};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename ProblemType>
struct ScanEnactor::Detail
{
	typedef ProblemType Problem;
	
	ScanEnactor *enactor;
	int max_grid_size;

	// Constructor
	Detail(ScanEnactor *enactor, int max_grid_size = 0) :
		enactor(enactor), max_grid_size(max_grid_size) {}
};


/**
 * Type for encapsulating storage details regarding an invocation
 */
template <typename T, typename SizeT>
struct ScanEnactor::Storage
{
	T *d_dest;
	T *d_src;
	SizeT num_elements;

	// Constructor
	Storage(T *d_dest, T *d_src, SizeT num_elements) :
		d_dest(d_dest), d_src(d_src), num_elements(num_elements) {}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
 */
template <scan::ProbSizeGenre PROB_SIZE_GENRE>
struct ScanEnactor::ConfigResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		typedef typename DetailType::Problem ProblemType;

		// Obtain tuned granularity type
		typedef scan::TunedConfig<ProblemType, CUDA_ARCH, PROB_SIZE_GENRE> TunedConfig;

		// Invoke base class enact with type
		return detail.enactor->template EnactInternal<TunedConfig, ScanEnactor>(
			storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
 */
template <>
struct ScanEnactor::ConfigResolver <scan::UNKNOWN>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		typedef typename DetailType::Problem ProblemType;

		// Obtain large tuned granularity type
		typedef scan::TunedConfig<ProblemType, CUDA_ARCH, scan::LARGE> LargeConfig;

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargeConfig::Upsweep::TILE_ELEMENTS *
			LargeConfig::Upsweep::CTA_OCCUPANCY *
			detail.enactor->SmCount();

		if (storage.num_elements < saturating_load) {

			// Invoke base class enact with small-problem config type
			typedef scan::TunedConfig<ProblemType, CUDA_ARCH, scan::SMALL> SmallConfig;
			return detail.enactor->template EnactInternal<SmallConfig, ScanEnactor>(
				storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
		}

		// Invoke base class enact with type
		return detail.enactor->template EnactInternal<LargeConfig, ScanEnactor>(
			storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
	}
};


/******************************************************************************
 * ScanEnactor Implementation
 ******************************************************************************/

/**
 * Performs a scan pass
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 */
template <typename TunedConfig>
cudaError_t ScanEnactor::EnactPass(
	typename TunedConfig::T *d_dest,
	typename TunedConfig::T *d_src,
	util::CtaWorkDistribution<typename TunedConfig::SizeT> &work,
	int spine_elements)
{
	using namespace scan;

	// Common problem type that can be used to reconstruct the same TunedConfig on the device
	typedef typename TunedConfig::ProblemType ProblemType;

	typedef typename TunedConfig::T T;
	typedef typename TunedConfig::Upsweep Upsweep;
	typedef typename TunedConfig::Spine Spine;
	typedef typename TunedConfig::Downsweep Downsweep;

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			TunedSpineKernel<ProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<1, TunedConfig::Spine::THREADS, 0>>>(
				d_src, d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor SpineKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option for dynamic smem allocation
			if (TunedConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, TunedUpsweepKernel<ProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"ScanEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, TunedSpineKernel<ProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"ScanEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, TunedDownsweepKernel<ProblemType, TunedConfig::PROB_SIZE_GENRE>),
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

			// Upsweep scan into spine
			TunedUpsweepKernel<ProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (T*) spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor TunedUpsweepKernel failed ", __FILE__, __LINE__))) break;

			// Spine scan
			TunedSpineKernel<ProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(T*) spine(), (T*) spine(), spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor TunedSpineKernel failed ", __FILE__, __LINE__))) break;

			// Downsweep scan into spine
			TunedDownsweepKernel<ProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
				d_src, d_dest, (T*) spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor TunedDownsweepKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}


/**
 * Enacts a scan operation on the specified device data using the
 * enumerated tuned granularity configuration
 */
template <
	typename T,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity(),
	scan::ProbSizeGenre PROB_SIZE_GENRE,
	typename SizeT>
cudaError_t ScanEnactor::Enact(
	T *d_dest,
	T *d_src,
	SizeT num_elements,
	int max_grid_size)
{
	typedef scan::ProblemType<T, SizeT, EXCLUSIVE, BinaryOp, Identity> Problem;
	typedef Detail<Problem> Detail;
	typedef Storage<T, SizeT> Storage;
	typedef ConfigResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, max_grid_size);
	Storage storage(d_dest, d_src, num_elements);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(storage, detail, PtxVersion());
}


/**
 * Enacts a scan operation on the specified device data using the
 * LARGE granularity configuration
 */
template <
	typename T,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity(),
	typename SizeT>
cudaError_t ScanEnactor::Enact(
	T *d_dest,
	T *d_src,
	SizeT num_elements,
	int max_grid_size)
{
	return Enact<T, EXCLUSIVE, BinaryOp, Identity, scan::UNKNOWN>(
		d_dest, d_src, num_elements, max_grid_size);
}



} // namespace b40c

