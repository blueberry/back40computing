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

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/segmented_scan/downsweep_kernel.cuh>
#include <b40c/segmented_scan/spine_kernel.cuh>
#include <b40c/segmented_scan/upsweep_kernel.cuh>

namespace b40c {
namespace segmented_scan {


/******************************************************************************
 * Enactor Declaration
 ******************************************************************************/

/**
 * Basic segmented scan enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced by separate CTAs
	util::Spine partial_spine;

	// Temporary device storage needed for reducing partials produced by separate CTAs
	util::Spine flag_spine;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Performs any lazy initialization work needed for this problem type
	 */
	template <typename ProblemConfig>
	cudaError_t Setup(int sweep_grid_size, int spine_elements);

    /**
	 * Performs a segmented scan pass
	 */
	template <typename ProblemConfig>
	cudaError_t EnactPass(
		typename ProblemConfig::T *d_dest,
		typename ProblemConfig::T *d_src,
		typename ProblemConfig::Flag *d_flag_src,
		util::CtaWorkDistribution<typename ProblemConfig::SizeT> &work,
		typename ProblemConfig::SizeT spine_elements);

	/**
	 * Enacts a segmented scan on the specified device data.
	 */
	template <typename ProblemConfig, typename EnactorType>
	cudaError_t EnactInternal(
		typename ProblemConfig::T *d_dest,
		typename ProblemConfig::T *d_src,
		typename ProblemConfig::Flag *d_flag_src,
		typename ProblemConfig::SizeT num_elements,
		int max_grid_size);

public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a segmented scan on the specified device data.
	 *
	 * For generating segmented scan kernels having computational granularities in accordance
	 * with user-supplied granularity-specialization types.  (Useful for auto-tuning.)
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
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename ProblemConfig>
	cudaError_t Enact(
		typename ProblemConfig::T *d_dest,
		typename ProblemConfig::T *d_src,
		typename ProblemConfig::Flag *d_flag_src,
		typename ProblemConfig::SizeT num_elements,
		int max_grid_size = 0);
};




/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/


/**
 * Performs any lazy initialization work needed for this problem type
 */
template <typename ProblemConfig>
cudaError_t Enactor::Setup(int sweep_grid_size, int spine_elements)
{
	typedef typename ProblemConfig::T T;
	typedef typename ProblemConfig::Flag Flag;

	cudaError_t retval = cudaSuccess;
	do {
		// Make sure our spines are big enough
		if (retval = partial_spine.Setup<T>(sweep_grid_size, spine_elements)) break;
		if (retval = flag_spine.Setup<Flag>(sweep_grid_size, spine_elements)) break;

	} while (0);

	return retval;
}


/**
 * Performs a segmented scan pass
 */
template <typename ProblemConfig>
cudaError_t Enactor::EnactPass(
	typename ProblemConfig::T *d_dest,
	typename ProblemConfig::T *d_src,
	typename ProblemConfig::Flag *d_flag_src,
	util::CtaWorkDistribution<typename ProblemConfig::SizeT> &work,
	typename ProblemConfig::SizeT spine_elements)
{
	typedef typename ProblemConfig::Upsweep Upsweep;
	typedef typename ProblemConfig::Spine Spine;
	typedef typename ProblemConfig::Downsweep Downsweep;
	typedef typename ProblemConfig::Single Single;
	typedef typename ProblemConfig::T T;
	typedef typename ProblemConfig::Flag Flag;

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			SpineKernel<Single><<<1, Single::THREADS, 0>>>(
				d_src, d_flag_src, d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option for dynamic smem allocation
			if (ProblemConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepKernel<Upsweep>),
					"Enactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineKernel<Spine>),
					"Enactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, DownsweepKernel<Downsweep>),
					"Enactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(
					upsweep_kernel_attrs.sharedSizeBytes,
					B40C_MAX(spine_kernel_attrs.sharedSizeBytes, downsweep_kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (ProblemConfig::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep segmented scan into spine
			UpsweepKernel<Upsweep><<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, d_flag_src, (T*) partial_spine(), (Flag*) flag_spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__))) break;

			// Spine segmented scan
			SpineKernel<Spine><<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(T*) partial_spine(), (Flag*) flag_spine(), (T*) partial_spine(), spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;

			// Downsweep segmented scan into spine
			DownsweepKernel<Downsweep><<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
				d_src, d_flag_src, d_dest, (T*) partial_spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}


/**
 * Enacts a segmented scan on the specified device data.
 */
template <typename ProblemConfig, typename EnactorType>
cudaError_t Enactor::EnactInternal(
	typename ProblemConfig::T *d_dest,
	typename ProblemConfig::T *d_src,
	typename ProblemConfig::Flag *d_flag_src,
	typename ProblemConfig::SizeT num_elements,
	int max_grid_size)
{
	typedef typename ProblemConfig::Upsweep Upsweep;
	typedef typename ProblemConfig::Spine Spine;
	typedef typename ProblemConfig::Downsweep Downsweep;
	typedef typename ProblemConfig::T T;
	typedef typename ProblemConfig::Flag Flag;
	typedef typename ProblemConfig::SizeT SizeT;

	// Compute sweep grid size
	const int MIN_OCCUPANCY = B40C_MIN(Downsweep::CTA_OCCUPANCY, Downsweep::CTA_OCCUPANCY);
	util::SuppressUnusedConstantWarning(MIN_OCCUPANCY);

	int sweep_grid_size = (ProblemConfig::OVERSUBSCRIBED_GRID_SIZE) ?
		OversubscribedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size) :
		OccupiedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size);

	if (num_elements <= Spine::TILE_ELEMENTS * 3) {
		// No need to upsweep reduce or downsweep segmented scan if we can do it
		// with a single spine kernel in three or less sequential
		// tiles (i.e., instead of three back-to-back tiles where we would
		// do one tile per up/spine/down kernel)
		sweep_grid_size = 1;
	}

	// Compute spine elements (round up to nearest spine tile_elements)
	int spine_elements = ((sweep_grid_size + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) * Spine::TILE_ELEMENTS;

	// Obtain a CTA work distribution for copying items of type T
	util::CtaWorkDistribution<SizeT> work;
	work.template Init<Upsweep::LOG_SCHEDULE_GRANULARITY>(num_elements, sweep_grid_size);

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
			printf("Work: \t\t[element bytes: %lu, flag bytes: %lu, SizeT %lu bytes, num_elements: %lu, schedule_granularity: %d, total_grains: %lu, grains_per_cta: %lu, extra_grains: %lu]\n",
				(unsigned long) sizeof(T), (unsigned long) sizeof(Flag), (unsigned long) sizeof(SizeT), (unsigned long) work.num_elements, Downsweep::SCHEDULE_GRANULARITY, (unsigned long) work.total_grains, (unsigned long) work.grains_per_cta, (unsigned long) work.extra_grains);
		} else {
			printf("Spine: \t\t[threads: %d, tile_elements: %d]\n",
				Spine::THREADS, Spine::TILE_ELEMENTS);
		}
	}

	cudaError_t retval = cudaSuccess;
	do {
		// Perform any lazy initialization work
		if (retval = Setup<ProblemConfig>(sweep_grid_size, spine_elements)) break;

		// Invoke segmented scan kernel
		EnactorType *dipatch = static_cast<EnactorType *>(this);
		if (retval = dipatch->template EnactPass<ProblemConfig>(
			d_dest,
			d_src,
			d_flag_src,
			work,
			spine_elements)) break;

	} while (0);

	return retval;
}


/**
 * Enacts a segmented scan on the specified device data.
 */
template <typename ProblemConfig>
cudaError_t Enactor::Enact(
	typename ProblemConfig::T *d_dest,
	typename ProblemConfig::T *d_src,
	typename ProblemConfig::Flag *d_flag_src,
	typename ProblemConfig::SizeT num_elements,
	int max_grid_size)
{
	return EnactInternal<ProblemConfig, Enactor>(
		d_dest,
		d_src,
		d_flag_src,
		num_elements,
		max_grid_size);
}


} // namespace segmented_scan
} // namespace b40c

