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
 * Tuned Scan Granularity Types
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>

#include <b40c/reduction/kernel_upsweep.cuh>
#include <b40c/scan/kernel_spine.cuh>
#include <b40c/scan/kernel_downsweep.cuh>
#include <b40c/scan/granularity.cuh>


namespace b40c {
namespace scan {


/******************************************************************************
 * Tuning classifiers to specialize granularity types on
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProbSizeGenre
{
	UNKNOWN = -1,			// Not actually specialized on: the enactor should use heuristics to select another size genre
	SMALL,
	LARGE
};


/**
 * Enumeration of architecture-families that we have tuned for below
 */
enum ArchFamily
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Classifies a given CUDA_ARCH into an architecture-family
 */
template <int CUDA_ARCH>
struct ArchFamilyClassifier
{
	static const ArchFamily FAMILY =	(CUDA_ARCH < SM13) ? 	SM10 :
										(CUDA_ARCH < SM20) ? 	SM13 :
																SM20;
};


/******************************************************************************
 * Granularity tuning types
 *
 * Specialized by family (and optionally by specific architecture) and by
 * problem size type
 *******************************************************************************/

/**
 * Granularity parameterization type
 */
template <
	typename ScanProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE,
	typename T = typename ScanProblemType::T,
	int T_SIZE = sizeof(typename ScanProblemType::T)>
struct TunedConfig;


/**
 * Default, catch-all granularity parameterization type.  Defers to the
 * architecture "family" that we know we have specialization type(s) for below.
 */
template <typename ScanProblemType, int CUDA_ARCH, ProbSizeGenre PROB_SIZE_GENRE, typename T, int T_SIZE>
struct TunedConfig : TunedConfig<
	ScanProblemType,
	ArchFamilyClassifier<CUDA_ARCH>::FAMILY,
	PROB_SIZE_GENRE,
	T,
	T_SIZE> {};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// All problems
template <typename ScanProblemType, ProbSizeGenre _PROB_SIZE_GENRE, typename T, int T_SIZE>
struct TunedConfig<ScanProblemType, SM20, _PROB_SIZE_GENRE, T, T_SIZE>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, false, false, 8, 8, 7, 1, 0, 5, 8, 0, 1, 5, 8, 7, 1, 0, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = _PROB_SIZE_GENRE;
};

// Large problems, 2B data
template <typename ScanProblemType, typename T>
struct TunedConfig<ScanProblemType, SM20, LARGE, T, 2>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, true, false, 10, 8, 7, 1, 2, 5, 8, 0, 1, 5, 8, 6, 2, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Large problems, 4B data
template <typename ScanProblemType, typename T>
struct TunedConfig<ScanProblemType, SM20, LARGE, T, 4>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, false, false, 11, 3, 9, 2, 0, 5, 8, 0, 1, 5, 3, 9, 1, 1, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Large problems, 8B data
template <typename ScanProblemType, typename T>
struct TunedConfig<ScanProblemType, SM20, LARGE, T, 8>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, true, false, 10, 8, 7, 1, 2, 5, 8, 0, 1, 5, 8, 6, 0, 2, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};



//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// All problems
template <typename ScanProblemType, ProbSizeGenre _PROB_SIZE_GENRE, typename T, int T_SIZE>
struct TunedConfig<ScanProblemType, SM13, _PROB_SIZE_GENRE, T, T_SIZE>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, false, false, 8, 8, 7, 1, 0, 5, 8, 0, 1, 5, 8, 7, 1, 0, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = _PROB_SIZE_GENRE;
};



//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------


// All problems
template <typename ScanProblemType, ProbSizeGenre _PROB_SIZE_GENRE, typename T, int T_SIZE>
struct TunedConfig<ScanProblemType, SM10, _PROB_SIZE_GENRE, T, T_SIZE>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, false, false, 8, 8, 7, 1, 0, 5, 8, 0, 1, 5, 8, 7, 1, 0, 5>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = _PROB_SIZE_GENRE;
};





/******************************************************************************
 * Scan kernel entry points that can derive a tuned granularity type
 * implicitly from the PROB_SIZE_GENRE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned upsweep reduction kernel entry point
 */
template <typename ScanProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ScanConfig::Upsweep::THREADS),
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ScanConfig::Upsweep::CTA_OCCUPANCY))
__global__ void TunedUpsweepReductionKernel(
	typename ScanProblemType::T 			*d_in,
	typename ScanProblemType::T 			*d_spine,
	typename ScanProblemType::SizeT 		* __restrict d_work_progress,
	util::CtaWorkDistribution<typename ScanProblemType::SizeT> work_decomposition,
	int progress_selector)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep ReductionKernelConfig;

	typename ScanProblemType::T *d_spine_partial = d_spine + blockIdx.x;

	reduction::UpsweepReductionPass<ReductionKernelConfig, ReductionKernelConfig::WORK_STEALING>::Invoke(
		d_in,
		d_spine_partial,
		d_work_progress,
		work_decomposition,
		progress_selector);
}

/**
 * Tuned spine scan kernel entry point
 */
template <typename ScanProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ScanConfig::Spine::THREADS),
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ScanConfig::Spine::CTA_OCCUPANCY))
__global__ void TunedSpineScanKernel(
	typename ScanProblemType::T 		* d_in,
	typename ScanProblemType::T 		* d_out,
	typename ScanProblemType::SizeT 	spine_elements)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine ScanKernelConfig;

	SpineScanPass<ScanKernelConfig>(d_in, d_out, spine_elements);
}


/**
 * Tuned downsweep scan kernel entry point
 */
template <typename ScanProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ScanConfig::Downsweep::THREADS),
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ScanConfig::Downsweep::CTA_OCCUPANCY))
__global__ void TunedDownsweepScanKernel(
	typename ScanProblemType::T 			* d_in,
	typename ScanProblemType::T 			* d_out,
	typename ScanProblemType::T 			* __restrict d_spine,
	util::CtaWorkDistribution<typename ScanProblemType::SizeT> work_decomposition)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep ScanKernelConfig;

	typename ScanProblemType::T *d_spine_partial = d_spine + blockIdx.x;

	DownsweepScanPass<ScanKernelConfig>(d_in, d_out, d_spine_partial, work_decomposition);
}






}// namespace scan
}// namespace b40c

