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

#include <b40c/scan/kernel_spine.cuh>
#include <b40c/scan/kernel_upsweep.cuh>
#include <b40c/scan/granularity.cuh>

namespace b40c {
namespace scan {


/******************************************************************************
 * Tuning classifiers to specialize granularity types on
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProblemSize
{
	SMALL 	= 0,
	LARGE 	= 1
};


/**
 * Enumeration of architecture-families that we have tuned for below
 */
enum Family
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Classifies a given CUDA_ARCH into an architecture-family
 */
template <int CUDA_ARCH>
struct FamilyClassifier
{
	static const Family FAMILY =	(CUDA_ARCH < SM13) ? 	SM10 :
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
	ProblemSize PROBLEM_SIZE,
	typename T = typename ScanProblemType::T,
	int T_SIZE = sizeof(typename ScanProblemType::T)>
struct TunedConfig;


/**
 * Default, catch-all granularity parameterization type.  Defers to the
 * architecture "family" that we know we have specialization type(s) for below.
 */
template <typename ScanProblemType, int CUDA_ARCH, ProblemSize PROBLEM_SIZE, typename T, int T_SIZE>
struct TunedConfig : TunedConfig<
	ScanProblemType,
	FamilyClassifier<CUDA_ARCH>::FAMILY,
	PROBLEM_SIZE,
	T,
	T_SIZE> {};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// All problems
template <typename ScanProblemType, ProblemSize _PROBLEM_SIZE, typename T, int T_SIZE>
struct TunedConfig<ScanProblemType, SM20, _PROBLEM_SIZE, T, T_SIZE>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, false, false, 8, 8, 7, 1, 0, 5, 8, 0, 1, 5, 8, 7, 1, 0, 5>
{
	static const ProblemSize PROBLEM_SIZE = _PROBLEM_SIZE;
};


//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// All problems
template <typename ScanProblemType, ProblemSize _PROBLEM_SIZE, typename T, int T_SIZE>
struct TunedConfig<ScanProblemType, SM13, _PROBLEM_SIZE, T, T_SIZE>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, false, false, 8, 8, 7, 1, 0, 5, 8, 0, 1, 5, 8, 7, 1, 0, 5>
{
	static const ProblemSize PROBLEM_SIZE = _PROBLEM_SIZE;
};



//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------


// All problems
template <typename ScanProblemType, ProblemSize _PROBLEM_SIZE, typename T, int T_SIZE>
struct TunedConfig<ScanProblemType, SM10, _PROBLEM_SIZE, T, T_SIZE>
	: ScanConfig<ScanProblemType, util::ld::NONE, util::st::NONE, false, false, false, 8, 8, 7, 1, 0, 5, 8, 0, 1, 5, 8, 7, 1, 0, 5>
{
	static const ProblemSize PROBLEM_SIZE = _PROBLEM_SIZE;
};





/******************************************************************************
 * Scan kernel entry points that can derive a tuned granularity type
 * implicitly from the PROBLEM_SIZE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned downsweep scan kernel entry point
 */
template <typename ScanProblemType, int PROBLEM_SIZE>
__launch_bounds__ (
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ScanConfig::Downsweep::THREADS),
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ScanConfig::Downsweep::CTA_OCCUPANCY))
__global__ void TunedDownsweepScanKernel(
	typename ScanProblemType::T 			* d_in,
	typename ScanProblemType::T 			* d_out,
	typename ScanProblemType::T 			* __restrict d_spine,
	util::CtaWorkDistribution<typename ScanProblemType::SizeT> work_decomposition)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::Downsweep ScanKernelConfig;

	typename ScanProblemType::T *d_spine_partial = d_spine + blockIdx.x;

	DownsweepScanPass(d_in, d_out, d_spine_partial, work_decomposition);
}


/**
 * Tuned spine scan kernel entry point
 */
template <typename ScanProblemType, int PROBLEM_SIZE>
__launch_bounds__ (
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ScanConfig::Spine::THREADS),
	(TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ScanConfig::Spine::CTA_OCCUPANCY))
__global__ void TunedSpineScanKernel(
	typename ScanProblemType::T 		* __restrict 	d_spine,
	typename ScanProblemType::SizeT 					spine_elements)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ScanProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::Spine ScanKernelConfig;

	SpineScanPass<ScanKernelConfig>(d_spine, d_out, spine_elements);
}



}// namespace scan
}// namespace b40c

