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
 * Tuned Reduction Problem Granularity Configuration Types
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>
#include <b40c/util/work_progress.cuh>

#include <b40c/reduction/kernel_spine.cuh>
#include <b40c/reduction/kernel_upsweep.cuh>
#include <b40c/reduction/problem_config.cuh>
#include <b40c/reduction/problem_type.cuh>

namespace b40c {
namespace reduction {


/******************************************************************************
 * Tuning classifiers to specialize granularity types on
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProbSizeGenre
{
	UNKNOWN = -1,			// Not actually specialized on: the enactor should use heuristics to select another size genre
	SMALL,					// Tuned @ 128KB input
	LARGE					// Tuned @ 128MB input
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
	static const ArchFamily FAMILY =	//(CUDA_ARCH < SM13) ? 	SM10 :			// Have not yet tuned configs for SM10-11
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
	typename ProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE,
	typename T = typename ProblemType::T,
	int T_SIZE = 1 << util::Log2<sizeof(typename ProblemType::T)>::VALUE>		// Round up to the nearest arch subword
struct TunedConfig;


/**
 * Default, catch-all granularity parameterization type.  Defers to the
 * architecture "family" that we know we have specialization type(s) for below.
 */
template <typename ProblemType, int CUDA_ARCH, ProbSizeGenre PROB_SIZE_GENRE, typename T, int T_SIZE>
struct TunedConfig : TunedConfig<
	ProblemType,
	ArchFamilyClassifier<CUDA_ARCH>::FAMILY,
	PROB_SIZE_GENRE,
	T,
	T_SIZE> {};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems, 8B+ data
template <typename ProblemType, typename T, int T_SIZE>
struct TunedConfig<ProblemType, SM20, LARGE, T, T_SIZE>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, true, false, true, false,
	  8, 7, 0, 2, 9,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Large problems, 4B data
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM20, LARGE, T, 4>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, true, true, false, false,
	  8, 7, 1, 2, 10,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Large problems, 2B data
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM20, LARGE, T, 2>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, true, true, false, false,
	  8, 7, 2, 2, 11,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Large problems, 1B data
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM20, LARGE, T, 1>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, false, false, true, false,
	  8, 7, 2, 2, 11,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};




// Small problems 8B+
template <typename ProblemType, typename T, int T_SIZE>
struct TunedConfig<ProblemType, SM20, SMALL, T, T_SIZE>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, false, true, false, false,
	  8, 5, 2, 1, 8,
	  6, 0, 0>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL;
};

// Small problems, 4B
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM20, SMALL, T, 4>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, false, true, false, false,
	  8, 5, 2, 1, 8,
	  7, 0, 0>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL;
};

// Small problems, 2B
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM20, SMALL, T, 2>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, false, true, false, false,
	  8, 5, 2, 1, 8,
	  7, 0, 0>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL;
};

// Small problems, 1B
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM20, SMALL, T, 1>
	: ProblemConfig<ProblemType, SM20, util::ld::NONE, util::st::NONE, false, true, false, false,
	  8, 5, 2, 1, 8,
	  7, 0, 0>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL;
};



//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// Large problems, 4B+
template <typename ProblemType, typename T, int T_SIZE>
struct TunedConfig<ProblemType, SM13, LARGE, T, T_SIZE>
	: ProblemConfig<ProblemType, SM13, util::ld::NONE, util::st::NONE, false, false, false, false,
	  8, 6, 0, 2, 8,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Large problems, 2B
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM13, LARGE, T, 2>
	: ProblemConfig<ProblemType, SM13, util::ld::NONE, util::st::NONE, false, false, false, false,
	  8, 6, 1, 2, 9,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Large problems, 1B
template <typename ProblemType, typename T>
struct TunedConfig<ProblemType, SM13, LARGE, T, 1>
	: ProblemConfig<ProblemType, SM13, util::ld::NONE, util::st::NONE, false, false, false, false,
	  4, 8, 2, 2, 12,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE;
};

// Small problems
template <typename ProblemType, typename T, int T_SIZE>
struct TunedConfig<ProblemType, SM13, SMALL, T, T_SIZE>
	: ProblemConfig<ProblemType, SM13, util::ld::NONE, util::st::NONE, false, false, false, false,
	  8, 5, 0, 2, 7,
	  8, 0, 1>
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL;
};



//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------







/******************************************************************************
 * Reduction kernel entry points that can derive a tuned granularity type
 * implicitly from the PROB_SIZE_GENRE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned upsweep reduction kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ProblemConfig::Upsweep::THREADS),
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ProblemConfig::Upsweep::CTA_OCCUPANCY))
__global__ void TunedUpsweepReductionKernel(
	typename ProblemType::T 								*d_in,
	typename ProblemType::T 								*d_spine,
	util::CtaWorkDistribution<typename ProblemType::SizeT> 	work_decomposition,
	util::WorkProgress										work_progress)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<
		ProblemType,
		__B40C_CUDA_ARCH__,
		(ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep ReductionKernelConfig;

	UpsweepReductionPass<ReductionKernelConfig, ReductionKernelConfig::WORK_STEALING>::Invoke(
		d_in,
		d_spine,
		work_decomposition,
		work_progress);
}


/**
 * Tuned spine reduction kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ProblemConfig::Spine::THREADS),
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::ProblemConfig::Spine::CTA_OCCUPANCY))
__global__ void TunedSpineReductionKernel(
	typename ProblemType::T 		*d_spine,
	typename ProblemType::T 		*d_out,
	typename ProblemType::SizeT 	spine_elements)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<
		ProblemType,
		__B40C_CUDA_ARCH__,
		(ProbSizeGenre) PROB_SIZE_GENRE>::Spine ReductionKernelConfig;

	SpineReductionPass<ReductionKernelConfig>(d_spine, d_out, spine_elements);
}



}// namespace reduction
}// namespace b40c

