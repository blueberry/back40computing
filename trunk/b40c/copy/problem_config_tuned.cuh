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
 * Tuned Copy Problem Granularity Configuration Types
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>

#include <b40c/copy/kernel_sweep.cuh>
#include <b40c/copy/problem_config.cuh>
#include <b40c/copy/problem_type.cuh>

namespace b40c {
namespace copy {


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
template <int CUDA_ARCH, ProbSizeGenre PROB_SIZE_GENRE>
struct TunedConfig;


/**
 * Default, catch-all granularity parameterization type.  Defers to the
 * architecture "family" that we know we have specialization type(s) for below.
 */
template <int CUDA_ARCH, ProbSizeGenre PROB_SIZE_GENRE>
struct TunedConfig : TunedConfig<
	ArchFamilyClassifier<CUDA_ARCH>::FAMILY,
	PROB_SIZE_GENRE> {};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

template <ProbSizeGenre _PROB_SIZE_GENRE>
struct TunedConfig<SM20, _PROB_SIZE_GENRE>
	: ProblemConfig<unsigned int, size_t, SM20, util::ld::CG, util::st::CG, true, false, 8, 7, 1, 1, 9>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = _PROB_SIZE_GENRE;
};

//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

template <ProbSizeGenre _PROB_SIZE_GENRE>
struct TunedConfig<SM13, _PROB_SIZE_GENRE>
	: ProblemConfig<unsigned int, size_t, SM13, util::ld::NONE, util::st::NONE, true, false, 8, 7, 1, 1, 9>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = _PROB_SIZE_GENRE;
};


//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------

template <ProbSizeGenre _PROB_SIZE_GENRE>
struct TunedConfig<SM10, _PROB_SIZE_GENRE>
	: ProblemConfig<unsigned int, size_t, SM10, util::ld::NONE, util::st::NONE, true, false, 8, 7, 1, 1, 9>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = _PROB_SIZE_GENRE;
};




/******************************************************************************
 * Copy kernel entry points that can derive a tuned granularity type
 * implicitly from the PROB_SIZE_GENRE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned byte-copy kernel entry point
 */
template <int PROB_SIZE_GENRE>
__launch_bounds__ (
	(TunedConfig<__B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ProblemConfig::Sweep::THREADS),
	(TunedConfig<__B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ProblemConfig::Sweep::CTA_OCCUPANCY))
__global__ void TunedSweepCopyKernel(
	void 								*d_in,
	void 								*d_out,
	size_t 								* __restrict d_work_progress,
	util::CtaWorkDistribution<size_t> 	work_decomposition,
	int									progress_selector,
	int 								extra_bytes)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<__B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Sweep CopyKernelConfig;
	typedef typename CopyKernelConfig::T T;

	T* out = (T*)(d_out);
	T* in = (T*)(d_in);

	SweepCopyPass<CopyKernelConfig, CopyKernelConfig::WORK_STEALING>::Invoke(
		in,
		out,
		d_work_progress,
		work_decomposition,
		progress_selector,
		extra_bytes);
}



}// namespace copy
}// namespace b40c

