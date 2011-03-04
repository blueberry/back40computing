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
 * Base Scan Enactor
 ******************************************************************************/

#pragma once

#include <b40c/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/scan/granularity.cuh>
#include <b40c/scan/kernel_downsweep.cuh>
#include <b40c/scan/kernel_spine.cuh>
#include <b40c/reduction/kernel_upsweep.cuh>

namespace b40c {


/******************************************************************************
 * ScanEnactor Declaration
 ******************************************************************************/

/**
 * Basic scan enactor class.
 */
template <typename DerivedEnactorType = void>
class ScanEnactor : public EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Dispatch type (either to self or to derived class as per CRTP -- curiously
	// recurring template pattern)
	typedef typename DispatchType<ScanEnactor, DerivedEnactorType>::Type Dispatch;
	Dispatch *dispatch;

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	void *d_spine;

	// Number of bytes backed by d_spine
	int spine_bytes;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Performs any lazy initialization work needed for this problem type
	 */
	template <typename ScanConfig>
	cudaError_t Setup(int sweep_grid_size, int spine_elements);

    /**
	 * Performs a scan pass
	 */
	template <typename ScanConfig>
	cudaError_t ScanPass(
		typename ScanConfig::Downsweep::T *d_dest,
		typename ScanConfig::Downsweep::T *d_src,
		util::CtaWorkDistribution<typename ScanConfig::Downsweep::SizeT> &work,
		typename ScanConfig::Spine::SizeT spine_elements);

	/**
	 * Dispatches upsweep kernel
	 */
	template <typename KernelConfig>
	cudaError_t DispatchUpsweep(
		int grid_size,
		int dynamic_smem,
		typename KernelConfig::T *d_src,
		typename KernelConfig::T *d_dest,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> &work);

	/**
	 * Dispatches spine kernel
	 */
	template <typename KernelConfig>
	cudaError_t DispatchSpine(
		int grid_size,
		int dynamic_smem,
		typename KernelConfig::T *d_src,
		typename KernelConfig::T *d_dest,
		typename KernelConfig::SizeT num_elements);

	/**
	 * Dispatches downsweep kernel
	 */
	template <typename KernelConfig>
	cudaError_t DispatchDownsweep(
		int grid_size,
		int dynamic_smem,
		typename KernelConfig::T *d_src,
		typename KernelConfig::T *d_dest,
		typename KernelConfig::T *d_spine,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> &work);


public:

	/**
	 * Constructor
	 */
	ScanEnactor();


	/**
     * Destructor
     */
    virtual ~ScanEnactor();


	/**
	 * Enacts a scan on the specified device data.
	 *
	 * For generating scan kernels having computational granularities in accordance
	 * with user-supplied granularity-specialization types.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be scanned
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename ScanConfig>
	cudaError_t Enact(
		typename ScanConfig::Downsweep::T *d_dest,
		typename ScanConfig::Downsweep::T *d_src,
		typename ScanConfig::Downsweep::SizeT num_elements,
		int max_grid_size = 0);
};




/******************************************************************************
 * ScanEnactor Implementation
 ******************************************************************************/

/**
 * Dispatches upsweep kernel
 */
template <typename DerivedEnactorType>
template <typename KernelConfig>
cudaError_t ScanEnactor<DerivedEnactorType>::DispatchUpsweep(
	int grid_size,
	int dynamic_smem,
	typename KernelConfig::T *d_src,
	typename KernelConfig::T *d_dest,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work)
{
	reduction::UpsweepReductionKernel<KernelConfig><<<grid_size, KernelConfig::THREADS, dynamic_smem>>>(
		d_src, d_dest, NULL, work, 0);

	return util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor UpsweepReductionKernel failed ", __FILE__, __LINE__);
}

/**
 * Dispatches spine kernel
 */
template <typename DerivedEnactorType>
template <typename KernelConfig>
cudaError_t ScanEnactor<DerivedEnactorType>::DispatchSpine(
	int grid_size,
	int dynamic_smem,
	typename KernelConfig::T *d_src,
	typename KernelConfig::T *d_dest,
	typename KernelConfig::SizeT num_elements)
{
	scan::SpineScanKernel<KernelConfig><<<grid_size, KernelConfig::THREADS, dynamic_smem>>>(
		d_src, d_dest, num_elements);

	return util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor SpineScanKernel failed ", __FILE__, __LINE__);
}

/**
 * Dispatches downsweep kernel
 */
template <typename DerivedEnactorType>
template <typename KernelConfig>
cudaError_t ScanEnactor<DerivedEnactorType>::DispatchDownsweep(
	int grid_size,
	int dynamic_smem,
	typename KernelConfig::T *d_src,
	typename KernelConfig::T *d_dest,
	typename KernelConfig::T *d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work)
{
	scan::DownsweepScanKernel<KernelConfig><<<grid_size, KernelConfig::THREADS, dynamic_smem>>>(
		d_src, d_dest, d_spine, work);

	return util::B40CPerror(cudaThreadSynchronize(), "ScanEnactor Downsweep failed ", __FILE__, __LINE__);
}


/**
 * Performs any lazy initialization work needed for this problem type
 */
template <typename DerivedEnactorType>
template <typename ScanConfig>
cudaError_t ScanEnactor<DerivedEnactorType>::Setup(int sweep_grid_size, int spine_elements)
{
	cudaError_t retval = cudaSuccess;
	do {
		// Make sure our spine is big enough
		int problem_spine_bytes = spine_elements * sizeof(typename ScanConfig::Downsweep::T);
		if (problem_spine_bytes > spine_bytes) {
			if (d_spine) {
				if (retval = util::B40CPerror(cudaFree(d_spine),
					"ScanEnactor cudaFree d_spine failed", __FILE__, __LINE__)) break;
			}

			spine_bytes = problem_spine_bytes;

			if (retval = util::B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
				"ScanEnactor cudaMalloc d_spine failed", __FILE__, __LINE__)) break;
		}
	} while (0);

	return retval;
}


/**
 * Performs a scan pass
 */
template <typename DerivedEnactorType>
template <typename ScanConfig>
cudaError_t ScanEnactor<DerivedEnactorType>::ScanPass(
	typename ScanConfig::Downsweep::T *d_dest,
	typename ScanConfig::Downsweep::T *d_src,
	util::CtaWorkDistribution<typename ScanConfig::Downsweep::SizeT> &work,
	typename ScanConfig::Spine::SizeT spine_elements)
{
	typedef typename ScanConfig::Upsweep Upsweep;
	typedef typename ScanConfig::Spine Spine;
	typedef typename ScanConfig::Downsweep Downsweep;

	typedef typename Downsweep::T T;

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			// No need to upsweep reduce or downsweep scan if there's only one CTA in the sweep grid
			int dynamic_smem = 0;
			if (retval = DispatchSpine<Spine>(work.grid_size, dynamic_smem, d_src, d_dest, work.num_elements)) break;

		} else {

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option for dynamic smem allocation
			if (ScanConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, reduction::UpsweepReductionKernel<Upsweep>),
					"ScanEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, scan::SpineScanKernel<Spine>),
					"ScanEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, scan::DownsweepScanKernel<Downsweep>),
					"ScanEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(
					upsweep_kernel_attrs.sharedSizeBytes,
					B40C_MAX(spine_kernel_attrs.sharedSizeBytes, downsweep_kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (ScanConfig::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep scan into spine
			if (retval = dispatch->template DispatchUpsweep<Upsweep>(grid_size[0], dynamic_smem[0], d_src, (T*) d_spine, work)) break;

			// Spine scan
			if (retval = dispatch->template DispatchSpine<Spine>(grid_size[1], dynamic_smem[1], (T*) d_spine, (T*) d_spine, spine_elements)) break;

			// Downsweep scan into spine
			if (retval = dispatch->template DispatchDownsweep<Downsweep>(grid_size[2], dynamic_smem[2], d_src, d_dest, (T*) d_spine, work)) break;

		}
	} while (0);

	return retval;
}


/**
 * Constructor
 */
template <typename DerivedEnactorType>
ScanEnactor<DerivedEnactorType>::ScanEnactor() :
	d_spine(NULL),
	spine_bytes(0)
{
	dispatch = static_cast<Dispatch*>(this);
}


/**
 * Destructor
 */
template <typename DerivedEnactorType>
ScanEnactor<DerivedEnactorType>::~ScanEnactor()
{
	if (d_spine) {
		util::B40CPerror(cudaFree(d_spine), "ScanEnactor cudaFree d_spine failed: ", __FILE__, __LINE__);
	}
}

    
/**
 * Enacts a scan on the specified device data.
 */
template <typename DerivedEnactorType>
template <typename ScanConfig>
cudaError_t ScanEnactor<DerivedEnactorType>::Enact(
	typename ScanConfig::Downsweep::T *d_dest,
	typename ScanConfig::Downsweep::T *d_src,
	typename ScanConfig::Downsweep::SizeT num_elements,
	int max_grid_size)
{
	typedef typename ScanConfig::Upsweep Upsweep;
	typedef typename ScanConfig::Spine Spine;
	typedef typename ScanConfig::Downsweep Downsweep;
	typedef typename Downsweep::T T;
	typedef typename Downsweep::SizeT SizeT;

	// Compute sweep grid size
	const int MIN_OCCUPANCY = B40C_MIN(Downsweep::CTA_OCCUPANCY, Downsweep::CTA_OCCUPANCY);
	util::SuppressUnusedConstantWarning(MIN_OCCUPANCY);
	int sweep_grid_size = (num_elements <= Spine::TILE_ELEMENTS) ?
		1 :
		(ScanConfig::OVERSUBSCRIBED_GRID_SIZE) ?
			OversubscribedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size) :
			OccupiedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size);

	// Compute spine elements (round up to nearest spine tile_elements)
	int spine_elements = ((sweep_grid_size + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) * Spine::TILE_ELEMENTS;

	// Obtain a CTA work distribution for copying items of type T
	util::CtaWorkDistribution<SizeT> work(num_elements, Downsweep::SCHEDULE_GRANULARITY, sweep_grid_size);

	if (DEBUG) {
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
		if (sweep_grid_size > 1) {
			printf("Upsweep: \t[sweep_grid_size: %d, threads %d, tile_elements: %d]\n",
				work.grid_size, Upsweep::THREADS, Upsweep::TILE_ELEMENTS);
			printf("Spine: \t\t[threads: %d, spine_elements: %d, tile_elements: %d]\n",
				Spine::THREADS, spine_elements, Spine::TILE_ELEMENTS);
			printf("Downsweep: \t[sweep_grid_size: %d, threads %d, tile_elements: %d]\n",
				work.grid_size, Downsweep::THREADS, Downsweep::TILE_ELEMENTS);
			printf("Work: \t\t[element bytes: %zu, SizeT %zu bytes, num_elements: %zu, schedule_granularity: %d, total_grains: %zu, grains_per_cta: %zu, extra_grains: %zu]\n",
				sizeof(T), sizeof(SizeT), work.num_elements, Downsweep::SCHEDULE_GRANULARITY, work.total_grains, work.grains_per_cta, work.extra_grains);
		} else {
			printf("Spine: \t\t[threads: %d, tile_elements: %d]\n",
				Spine::THREADS, Spine::TILE_ELEMENTS);
		}
	}

	cudaError_t retval = cudaSuccess;
	do {
		// Perform any lazy initialization work
		if (retval = Setup<ScanConfig>(sweep_grid_size, spine_elements)) break;

		// Invoke scan kernel
		if (retval = dispatch->template ScanPass<ScanConfig>(d_dest, d_src, work, spine_elements)) break;

	} while (0);

	return retval;
}



}// namespace b40c

