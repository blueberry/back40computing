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
 * Autotuned radix sort policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/radix_sort/upsweep/kernel.cuh>
#include <b40c/radix_sort/upsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel.cuh>
#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/policy.cuh>

#include <b40c/scan/spine/kernel.cuh>

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Genre classifiers to specialize policy types on
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProbSizeGenre
{
	UNKNOWN_SIZE = -1,			// Not actually specialized on: the enactor should use heuristics to select another size genre
	SMALL_SIZE,					// Tuned @ 128KB input
	LARGE_SIZE					// Tuned @ 128MB input
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
struct ArchGenre
{
	static const ArchFamily FAMILY =	(CUDA_ARCH < SM13) ? 	SM10 :
										(CUDA_ARCH < SM20) ? 	SM13 :
																SM20;
};


/**
 * Classifies data type size into small and large
 */
template <typename ProblemType>
struct TypeSizeGenre
{
	enum {
		SMALL_TYPE = (B40C_MAX(sizeof(typename ProblemType::KeyType), sizeof(typename ProblemType::ValueType)) <= 4),
	};
};


/**
 * Classifies pointer type size into small and large
 */
template <typename ProblemType>
struct PointerSizeGenre
{
	enum {
		SMALL_TYPE = (B40C_MAX(sizeof(typename ProblemType::SizeT), sizeof(size_t)) <= 4),
	};
};


/**
 * Autotuning policy genre, to be specialized
 */
template <
	typename ProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedGenre :
	AutotunedGenre<ProblemType, ArchGenre<CUDA_ARCH>::FAMILY, PROB_SIZE_GENRE>
{};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------



// Large problems
template <typename ProblemType>
struct AutotunedGenre<ProblemType, SM20, LARGE_SIZE> : Policy<
	// Problem Type
	ProblemType,

	// Common
	SM20,
	4,						// RADIX_BITS
	10,						// LOG_SCHEDULE_GRANULARITY
	util::io::ld::NONE,		// CACHE_MODIFIER
	util::io::st::NONE,		// CACHE_MODIFIER
	true,					// EARLY_EXIT
	false,					// UNIFORM_SMEM_ALLOCATION
	true, 					// UNIFORM_GRID_SIZE
	true,					// OVERSUBSCRIBED_GRID_SIZE

	// Upsweep Kernel
	8,						// UPSWEEP_CTA_OCCUPANCY
	7,						// UPSWEEP_LOG_THREADS
	0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
	2,						// UPSWEEP_LOG_LOADS_PER_TILE

	// Spine-scan Kernel
	1,						// SPINE_CTA_OCCUPANCY
	7,						// SPINE_LOG_THREADS
	2,						// SPINE_LOG_LOAD_VEC_SIZE
	0,						// SPINE_LOG_LOADS_PER_TILE
	5,						// SPINE_LOG_RAKING_THREADS

	// Downsweep Kernel
	true,					// DOWNSWEEP_TWO_PHASE_SCATTER
	8,						// DOWNSWEEP_CTA_OCCUPANCY
	6,						// DOWNSWEEP_LOG_THREADS
	2,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
	1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
	(TypeSizeGenre<ProblemType>::SMALL_TYPE && PointerSizeGenre<ProblemType>::SMALL_TYPE) ?		// DOWNSWEEP_LOG_CYCLES_PER_TILE
		1 :
		0,
	6>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};


// Small problems
template <typename ProblemType>
struct AutotunedGenre<ProblemType, SM20, SMALL_SIZE> : Policy<
	// Problem Type
	ProblemType,

	// Common
	SM20,
	4,						// RADIX_BITS
	9,						// LOG_SCHEDULE_GRANULARITY
	util::io::ld::NONE,		// CACHE_MODIFIER
	util::io::st::NONE,		// CACHE_MODIFIER
	false,					// EARLY_EXIT
	false,					// UNIFORM_SMEM_ALLOCATION
	false, 					// UNIFORM_GRID_SIZE
	false,					// OVERSUBSCRIBED_GRID_SIZE

	// Upsweep Kernel
	8,						// UPSWEEP_CTA_OCCUPANCY
	7,						// UPSWEEP_LOG_THREADS
	1,						// UPSWEEP_LOG_LOAD_VEC_SIZE
	0,						// UPSWEEP_LOG_LOADS_PER_TILE

	// Spine-scan Kernel
	1,						// SPINE_CTA_OCCUPANCY
	8,						// SPINE_LOG_THREADS
	2,						// SPINE_LOG_LOAD_VEC_SIZE
	0,						// SPINE_LOG_LOADS_PER_TILE
	5,						// SPINE_LOG_RAKING_THREADS

	// Downsweep Kernel
	true,					// DOWNSWEEP_TWO_PHASE_SCATTER
	7,						// DOWNSWEEP_CTA_OCCUPANCY
	7,						// DOWNSWEEP_LOG_THREADS
	1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
	1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
	0, 						// DOWNSWEEP_LOG_CYCLES_PER_TILE
	7>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};


//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <typename ProblemType>
struct AutotunedGenre<ProblemType, SM13, LARGE_SIZE> : Policy<
	// Problem Type
	ProblemType,

	// Common
	SM13,
	4,						// RADIX_BITS
	9,						// LOG_SCHEDULE_GRANULARITY
	util::io::ld::NONE,		// CACHE_MODIFIER
	util::io::st::NONE,		// CACHE_MODIFIER
	true,					// EARLY_EXIT
	true,					// UNIFORM_SMEM_ALLOCATION
	true, 					// UNIFORM_GRID_SIZE
	true,					// OVERSUBSCRIBED_GRID_SIZE

	// Upsweep Kernel
	5,						// UPSWEEP_CTA_OCCUPANCY
	7,						// UPSWEEP_LOG_THREADS
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// UPSWEEP_LOG_LOAD_VEC_SIZE
		1 :
		0,
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// UPSWEEP_LOG_LOADS_PER_TILE
		0 :
		1,

	// Spine-scan Kernel
	1,						// SPINE_CTA_OCCUPANCY
	7,						// SPINE_LOG_THREADS
	2,						// SPINE_LOG_LOAD_VEC_SIZE
	0,						// SPINE_LOG_LOADS_PER_TILE
	5,						// SPINE_LOG_RAKING_THREADS

	// Downsweep Kernel
	true,					// DOWNSWEEP_TWO_PHASE_SCATTER
	5,						// DOWNSWEEP_CTA_OCCUPANCY
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// DOWNSWEEP_LOG_THREADS
		6 :
		7,
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// DOWNSWEEP_LOG_LOAD_VEC_SIZE
		2 :
		1,
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// DOWNSWEEP_LOG_LOADS_PER_CYCLE
		1 :
		0,
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// DOWNSWEEP_LOG_CYCLES_PER_TILE
		0 :
		0,
	5>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};


// Small problems
template <typename ProblemType>
struct AutotunedGenre<ProblemType, SM13, SMALL_SIZE> : Policy<
	// Problem Type
	ProblemType,

	// Common
	SM13,
	4,						// RADIX_BITS
	9,						// LOG_SCHEDULE_GRANULARITY
	util::io::ld::NONE,		// CACHE_MODIFIER
	util::io::st::NONE,		// CACHE_MODIFIER
	true,					// EARLY_EXIT
	true,					// UNIFORM_SMEM_ALLOCATION
	true, 					// UNIFORM_GRID_SIZE
	true,					// OVERSUBSCRIBED_GRID_SIZE

	// Upsweep Kernel
	5,						// UPSWEEP_CTA_OCCUPANCY
	7,						// UPSWEEP_LOG_THREADS
	1,						// UPSWEEP_LOG_LOAD_VEC_SIZE
	0,						// UPSWEEP_LOG_LOADS_PER_TILE

	// Spine-scan Kernel
	1,						// SPINE_CTA_OCCUPANCY
	7,						// SPINE_LOG_THREADS
	2,						// SPINE_LOG_LOAD_VEC_SIZE
	0,						// SPINE_LOG_LOADS_PER_TILE
	5,						// SPINE_LOG_RAKING_THREADS

	// Downsweep Kernel
	true,					// DOWNSWEEP_TWO_PHASE_SCATTER
	5,						// DOWNSWEEP_CTA_OCCUPANCY
	6,						// DOWNSWEEP_LOG_THREADS
	2,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
	1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// DOWNSWEEP_LOG_CYCLES_PER_TILE
		0 :
		0,
	5>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};




//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <typename ProblemType>
struct AutotunedGenre<ProblemType, SM10, LARGE_SIZE> : Policy<
	// Problem Type
	ProblemType,

	// Common
	SM10,
	4,						// RADIX_BITS
	9,						// LOG_SCHEDULE_GRANULARITY
	util::io::ld::NONE,		// CACHE_MODIFIER
	util::io::st::NONE,		// CACHE_MODIFIER
	true,					// EARLY_EXIT
	false,					// UNIFORM_SMEM_ALLOCATION
	true, 					// UNIFORM_GRID_SIZE
	true,					// OVERSUBSCRIBED_GRID_SIZE

	// Upsweep Kernel
	3,						// UPSWEEP_CTA_OCCUPANCY
	7,						// UPSWEEP_LOG_THREADS
	0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
	0,						// UPSWEEP_LOG_LOADS_PER_TILE

	// Spine-scan Kernel
	1,						// SPINE_CTA_OCCUPANCY
	7,						// SPINE_LOG_THREADS
	2,						// SPINE_LOG_LOAD_VEC_SIZE
	0,						// SPINE_LOG_LOADS_PER_TILE
	5,						// SPINE_LOG_RAKING_THREADS

	// Downsweep Kernel
	true,					// DOWNSWEEP_TWO_PHASE_SCATTER
	2,						// DOWNSWEEP_CTA_OCCUPANCY
	7,						// DOWNSWEEP_LOG_THREADS
	1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
	1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?
		1 :
		1,
	7>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = LARGE_SIZE;
};


// Small problems
template <typename ProblemType>
struct AutotunedGenre<ProblemType, SM10, SMALL_SIZE> : Policy<
	// Problem Type
	ProblemType,

	// Common
	SM10,
	4,						// RADIX_BITS
	9,						// LOG_SCHEDULE_GRANULARITY
	util::io::ld::NONE,		// CACHE_MODIFIER
	util::io::st::NONE,		// CACHE_MODIFIER
	true,					// EARLY_EXIT
	false,					// UNIFORM_SMEM_ALLOCATION
	true, 					// UNIFORM_GRID_SIZE
	true,					// OVERSUBSCRIBED_GRID_SIZE

	// Upsweep Kernel
	3,						// UPSWEEP_CTA_OCCUPANCY
	7,						// UPSWEEP_LOG_THREADS
	0,						// UPSWEEP_LOG_LOAD_VEC_SIZE
	0,						// UPSWEEP_LOG_LOADS_PER_TILE

	// Spine-scan Kernel
	1,						// SPINE_CTA_OCCUPANCY
	7,						// SPINE_LOG_THREADS
	2,						// SPINE_LOG_LOAD_VEC_SIZE
	0,						// SPINE_LOG_LOADS_PER_TILE
	5,						// SPINE_LOG_RAKING_THREADS

	// Downsweep Kernel
	true,					// DOWNSWEEP_TWO_PHASE_SCATTER
	2,						// DOWNSWEEP_CTA_OCCUPANCY
	7,						// DOWNSWEEP_LOG_THREADS
	1,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE
	1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE
	(TypeSizeGenre<ProblemType>::SMALL_TYPE) ?	// DOWNSWEEP_LOG_CYCLES_PER_TILE
		1 :
		1,
	7>						// DOWNSWEEP_LOG_RAKING_THREADS
{
	static const ProbSizeGenre PROB_SIZE_GENRE = SMALL_SIZE;
};




/******************************************************************************
 * Kernel entry points that can derive a tuned granularity type
 * implicitly from the PROB_SIZE_GENRE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned upsweep reduction kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE, typename PassPolicy>
__launch_bounds__ (
	(AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep::THREADS),
	(AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep::MAX_CTA_OCCUPANCY))
__global__ void TunedUpsweepKernel(
	int 													*d_selectors,
	typename ProblemType::SizeT 							*d_spine,
	typename ProblemType::KeyType 							*d_in_keys,
	typename ProblemType::KeyType 							*d_out_keys,
	util::CtaWorkDistribution<typename ProblemType::SizeT> 	work_decomposition)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Upsweep TuningPolicy;
	typedef upsweep::KernelPolicy<TuningPolicy, PassPolicy> KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	upsweep::UpsweepPass<KernelPolicy>(
		d_selectors,
		d_spine,
		d_in_keys,
		d_out_keys,
		work_decomposition,
		smem_storage);
}

/**
 * Tuned spine scan kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE>
__launch_bounds__ (
	(AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine::THREADS),
	(AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine::CTA_OCCUPANCY))
__global__ void TunedSpineKernel(
	typename ProblemType::SizeT 		*d_spine_in,
	typename ProblemType::SizeT 		*d_spine_out,
	int									spine_elements)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Spine KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	// Invoke the wrapped kernel logic
	scan::spine::SpinePass<KernelPolicy>(d_spine_in, d_spine_out, spine_elements, smem_storage);
}


/**
 * Tuned downsweep scan kernel entry point
 */
template <typename ProblemType, int PROB_SIZE_GENRE, typename PassPolicy>
__launch_bounds__ (
	(AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep::THREADS),
	(AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep::MAX_CTA_OCCUPANCY))
__global__ void TunedDownsweepKernel(
	int 													*d_selectors,
	typename ProblemType::SizeT 							*d_spine,
	typename ProblemType::KeyType 							*d_keys0,
	typename ProblemType::KeyType 							*d_keys1,
	typename ProblemType::ValueType 						*d_values0,
	typename ProblemType::ValueType							*d_values1,
	util::CtaWorkDistribution<typename ProblemType::SizeT>	work_decomposition);

template <typename ProblemType, int PROB_SIZE_GENRE, typename PassPolicy>
__global__ void TunedDownsweepKernel(
	int 													*d_selectors,
	typename ProblemType::SizeT 							*d_spine,
	typename ProblemType::KeyType 							*d_keys0,
	typename ProblemType::KeyType 							*d_keys1,
	typename ProblemType::ValueType 						*d_values0,
	typename ProblemType::ValueType							*d_values1,
	util::CtaWorkDistribution<typename ProblemType::SizeT>	work_decomposition)
{
	// Load the kernel policy type identified by the enum for this architecture
	typedef typename AutotunedGenre<ProblemType, __B40C_CUDA_ARCH__, (ProbSizeGenre) PROB_SIZE_GENRE>::Downsweep TuningPolicy;
	typedef downsweep::KernelPolicy<TuningPolicy, PassPolicy> KernelPolicy;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	// Invoke the wrapped kernel logic
	downsweep::DownsweepPass<KernelPolicy>(
		d_selectors,
		d_spine,
		d_keys0,
		d_keys1,
		d_values0,
		d_values1,
		work_decomposition,
		smem_storage);
}


/******************************************************************************
 * Autotuned scan policy
 *******************************************************************************/



/**
 * Autotuned policy type, derives from autotuned genre
 */
template <
	typename ProblemType,
	int CUDA_ARCH,
	ProbSizeGenre PROB_SIZE_GENRE>
struct AutotunedPolicy :
	AutotunedGenre<
		ProblemType,
		CUDA_ARCH,
		PROB_SIZE_GENRE>
{
	typedef typename ProblemType::KeyType 		KeyType;
	typedef typename ProblemType::ValueType 	ValueType;
	typedef typename ProblemType::SizeT 		SizeT;

	typedef void (*UpsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(SizeT*, SizeT*, int);
	typedef void (*DownsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, ValueType*, ValueType*, util::CtaWorkDistribution<SizeT>);

	//---------------------------------------------------------------------
	// Kernel function pointer retrieval
	//---------------------------------------------------------------------

	template <typename PassPolicy>
	static UpsweepKernelPtr UpsweepKernel()
	{
		return TunedUpsweepKernel<ProblemType, PROB_SIZE_GENRE, PassPolicy>;
	}

	static SpineKernelPtr SpineKernel()
	{
		return TunedSpineKernel<ProblemType, PROB_SIZE_GENRE>;
	}

	template <typename PassPolicy>
	static DownsweepKernelPtr DownsweepKernel()
	{
		return TunedDownsweepKernel<ProblemType, PROB_SIZE_GENRE, PassPolicy>;
	}
};



}// namespace radix_sort
}// namespace b40c

