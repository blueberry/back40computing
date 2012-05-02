/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * Unified radix sort policy
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>


namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Pass policy
 ******************************************************************************/

/**
 * Pass policy
 */
template <

	// Common
	int TUNE_ARCH,
	typename ProblemType,
	int _CURRENT_BIT,
	int _RADIX_BITS,
	int _CURRENT_PASS,

	// Dispatch
	int LOG_SCHEDULE_GRANULARITY,
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	bool EARLY_EXIT,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	bool _OVERSUBSCRIBED_GRID_SIZE,

	// Upsweep
	int UPSWEEP_MIN_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,

	// Spine-scan
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS,

	// Downsweep
	partition::downsweep::ScatterStrategy DOWNSWEEP_SCATTER_STRATEGY,
	int DOWNSWEEP_MIN_CTA_OCCUPANCY,
	int DOWNSWEEP_LOG_THREADS,
	int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_LOG_LOADS_PER_TILE,
	int DOWNSWEEP_LOG_RAKING_THREADS>

struct PassPolicy
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename ProblemType::KeyType 		KeyType;
	typedef typename ProblemType::ValueType		ValueType;
	typedef typename ProblemType::SizeT 		SizeT;

	typedef void (*UpsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(SizeT*, SizeT*, int);
	typedef void (*DownsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, ValueType*, ValueType*, util::CtaWorkDistribution<SizeT>);

	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	enum {
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
		OVERSUBSCRIBED_GRID_SIZE	= _OVERSUBSCRIBED_GRID_SIZE,
		CHECK_ALIGNMENT 			= false,
	};

	//---------------------------------------------------------------------
	// Tuning Policies
	//---------------------------------------------------------------------

	// Upsweep
	typedef upsweep::KernelPolicy<
		ProblemType,
		TUNE_ARCH,
		CHECK_ALIGNMENT,
		RADIX_BITS,
		LOG_SCHEDULE_GRANULARITY,
		UPSWEEP_MIN_CTA_OCCUPANCY,
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		EARLY_EXIT>
			Upsweep;

	// Problem type for spine scan
	typedef scan::ProblemType<
		SizeT,								// spine scan type T
		int,								// spine scan SizeT
		util::Sum<SizeT>,
		util::Sum<SizeT>,
		true,								// exclusive
		true> SpineProblemType;				// addition is commutative

	// Kernel config for spine scan
	typedef scan::KernelPolicy <
		SpineProblemType,
		CUDA_ARCH,
		false,								// do not check alignment
		1,									// only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
			Spine;

	// Downsweep
	typedef downsweep::KernelPolicy<
		ProblemType,
		TUNE_ARCH,
		CHECK_ALIGNMENT,
		_RADIX_BITS,
		LOG_SCHEDULE_GRANULARITY,
		DOWNSWEEP_MIN_CTA_OCCUPANCY,
		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		DOWNSWEEP_SCATTER_STRATEGY,
		EARLY_EXIT>
			Downsweep;


};


/******************************************************************************
 * Tuned pass policy
 ******************************************************************************/

/**
 * Tuned pass policy
 */
template <
	int 		TUNE_ARCH,
	typename 	ProblemType,
	int 		CURRENT_BIT,
	int 		BITS_REMAINING,
	int 		CURRENT_PASS>
struct TunedPassPolicy;


/**
 * SM20
 */
template <
	typename ProblemType,
	int CURRENT_BIT,
	int BITS_REMAINING,
	int CURRENT_PASS>
struct TunedPassPolicy<200, ProblemType, CURRENT_BIT, BITS_REMAINING, CURRENT_PASS> :
	b40c::radix_sort::PassPolicy<

		// Common
		200,						// TUNE_ARCH
		ProblemType,				// Problem type
		CURRENT_BIT,				// Current bit
		CUB_MIN(BITS_REMAINING, 5),	// RADIX_BITS
		CURRENT_PASS,				// Current pass

		// Dispatch tuning policy
		12,							// LOG_SCHEDULE_GRANULARITY			The "grain" by which to divide up the problem input.  E.g., 7 implies a near-even distribution of 128-key chunks to each CTA.  Related to, but different from the upsweep/downswep tile sizes, which may be different from each other.
		b40c::util::io::ld::NONE,	// CACHE_MODIFIER					Load cache-modifier.  Valid values: NONE, ca, cg, cs
		b40c::util::io::st::NONE,	// CACHE_MODIFIER					Store cache-modifier.  Valid values: NONE, wb, cg, cs
		false,						// EARLY_EXIT						Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
		false,						// UNIFORM_SMEM_ALLOCATION			Whether or not to pad the dynamic smem allocation to ensure that all three kernels (upsweep, spine, downsweep) have the same overall smem allocation
		true, 						// UNIFORM_GRID_SIZE				Whether or not to launch the spine kernel with one CTA (all that's needed), or pad it up to the same grid size as the upsweep/downsweep kernels
		true,						// OVERSUBSCRIBED_GRID_SIZE			Whether or not to oversubscribe the GPU with CTAs, up to a constant factor (usually 4x the resident occupancy)

		// Upsweep kernel policy
		8,							// UPSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
		7,							// UPSWEEP_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		2,							// UPSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2
		1,							// UPSWEEP_LOG_LOADS_PER_TILE		The number of loads (log) per tile.  Valid range: 0-2

		// Spine-scan kernel policy
		8,							// SPINE_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		2,							// SPINE_LOG_LOAD_VEC_SIZE			The vector-load size (log) for each load (log).  Valid range: 0-2
		2,							// SPINE_LOG_LOADS_PER_TILE			The number of loads (log) per tile.  Valid range: 0-2
		5,							// SPINE_LOG_RAKING_THREADS			The number of raking threads (log) for local prefix sum.  Valid range: 5-SPINE_LOG_THREADS

		// Policy for downsweep kernel
		b40c::partition::downsweep::SCATTER_TWO_PHASE,			// DOWNSWEEP_TWO_PHASE_SCATTER		Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
		ProblemType::KEYS_ONLY ? 4 : 2,							// DOWNSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
		ProblemType::KEYS_ONLY ? 7 : 8,							// DOWNSWEEP_LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10, subject to constraints described above
		ProblemType::KEYS_ONLY ? 4 : 4,							// DOWNSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2, subject to constraints described above
		ProblemType::KEYS_ONLY ? 7 : 8>							// DOWNSWEEP_LOG_RAKING_THREADS		The number of raking threads (log) for local prefix sum.  Valid range: 5-DOWNSWEEP_LOG_THREADS
{};


/**
 * SM13
 */
template <
	typename ProblemType,
	int CURRENT_BIT,
	int BITS_REMAINING,
	int CURRENT_PASS>
struct TunedPassPolicy<130, ProblemType, CURRENT_BIT, BITS_REMAINING, CURRENT_PASS> :
	b40c::radix_sort::PassPolicy<

		// Common
		130,						// TUNE_ARCH
		ProblemType,				// Problem type
		CURRENT_BIT,				// Current bit
		CUB_MIN(BITS_REMAINING, 5),	// RADIX_BITS
		CURRENT_PASS,				// Current pass

		// Dispatch tuning policy
		10,							// LOG_SCHEDULE_GRANULARITY			The "grain" by which to divide up the problem input.  E.g., 7 implies a near-even distribution of 128-key chunks to each CTA.  Related to, but different from the upsweep/downswep tile sizes, which may be different from each other.
		b40c::util::io::ld::NONE,	// CACHE_MODIFIER					Load cache-modifier.  Valid values: NONE, ca, cg, cs
		b40c::util::io::st::NONE,	// CACHE_MODIFIER					Store cache-modifier.  Valid values: NONE, wb, cg, cs
		false,						// EARLY_EXIT						Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
		true,						// UNIFORM_SMEM_ALLOCATION			Whether or not to pad the dynamic smem allocation to ensure that all three kernels (upsweep, spine, downsweep) have the same overall smem allocation
		true, 						// UNIFORM_GRID_SIZE				Whether or not to launch the spine kernel with one CTA (all that's needed), or pad it up to the same grid size as the upsweep/downsweep kernels
		true,						// OVERSUBSCRIBED_GRID_SIZE			Whether or not to oversubscribe the GPU with CTAs, up to a constant factor (usually 4x the resident occupancy)

		// Upsweep kernel policy
		(KEY_BITS > 4) ?			// UPSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
			3 :							// 5bit
			6,							// 4bit
		7,							// UPSWEEP_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		0,							// UPSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2
		2,							// UPSWEEP_LOG_LOADS_PER_TILE		The number of loads (log) per tile.  Valid range: 0-2

		// Spine-scan kernel policy
		8,							// SPINE_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		2,							// SPINE_LOG_LOAD_VEC_SIZE			The vector-load size (log) for each load (log).  Valid range: 0-2
		0,							// SPINE_LOG_LOADS_PER_TILE			The number of loads (log) per tile.  Valid range: 0-2
		5,							// SPINE_LOG_RAKING_THREADS			The number of raking threads (log) for local prefix sum.  Valid range: 5-SPINE_LOG_THREADS

		// Downsweep kernel policy
		b40c::partition::downsweep::SCATTER_TWO_PHASE,			// DOWNSWEEP_TWO_PHASE_SCATTER		Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
			3 :
			2,
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10, subject to constraints described above
			6 :
			6,
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2, subject to constraints described above
			4 :
			4,
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_LOG_RAKING_THREADS		The number of raking threads (log) for local prefix sum.  Valid range: 5-DOWNSWEEP_LOG_THREADS
			6 :
			6>
{};


/**
 * SM10
 */
template <
	typename ProblemType,
	int CURRENT_BIT,
	int BITS_REMAINING,
	int CURRENT_PASS>
struct TunedPassPolicy<100, ProblemType, CURRENT_BIT, BITS_REMAINING, CURRENT_PASS> :
	b40c::radix_sort::PassPolicy<

		// Common
		100,						// TUNE_ARCH
		ProblemType,				// Problem type
		CURRENT_BIT,				// Current bit
		CUB_MIN(BITS_REMAINING, 5),	// RADIX_BITS
		CURRENT_PASS,				// Current pass

		// Dispatch tuning policy
		10,							// LOG_SCHEDULE_GRANULARITY			The "grain" by which to divide up the problem input.  E.g., 7 implies a near-even distribution of 128-key chunks to each CTA.  Related to, but different from the upsweep/downswep tile sizes, which may be different from each other.
		b40c::util::io::ld::NONE,	// CACHE_MODIFIER					Load cache-modifier.  Valid values: NONE, ca, cg, cs
		b40c::util::io::st::NONE,	// CACHE_MODIFIER					Store cache-modifier.  Valid values: NONE, wb, cg, cs
		false,						// EARLY_EXIT						Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
		false,						// UNIFORM_SMEM_ALLOCATION			Whether or not to pad the dynamic smem allocation to ensure that all three kernels (upsweep, spine, downsweep) have the same overall smem allocation
		true, 						// UNIFORM_GRID_SIZE				Whether or not to launch the spine kernel with one CTA (all that's needed), or pad it up to the same grid size as the upsweep/downsweep kernels
		true,						// OVERSUBSCRIBED_GRID_SIZE			Whether or not to oversubscribe the GPU with CTAs, up to a constant factor (usually 4x the resident occupancy)

		// Upsweep kernel policy
		(KEY_BITS > 4) ?			// UPSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
			2 :							// 5bit
			2,							// 4bit
		7,							// UPSWEEP_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		0,							// UPSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2
		0,							// UPSWEEP_LOG_LOADS_PER_TILE		The number of loads (log) per tile.  Valid range: 0-2

		// Spine-scan kernel policy
		7,							// SPINE_LOG_THREADS				The number of threads (log) to launch per CTA.  Valid range: 5-10
		2,							// SPINE_LOG_LOAD_VEC_SIZE			The vector-load size (log) for each load (log).  Valid range: 0-2
		0,							// SPINE_LOG_LOADS_PER_TILE			The number of loads (log) per tile.  Valid range: 0-2
		5,							// SPINE_LOG_RAKING_THREADS			The number of raking threads (log) for local prefix sum.  Valid range: 5-SPINE_LOG_THREADS

		// Downsweep kernel policy
		b40c::partition::downsweep::SCATTER_TWO_PHASE,			// DOWNSWEEP_TWO_PHASE_SCATTER		Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_CTA_OCCUPANCY			The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
			2 :
			2,
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10, subject to constraints described above
			6 :
			6,
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_LOG_LOAD_VEC_SIZE		The vector-load size (log) for each load (log).  Valid range: 0-2, subject to constraints described above
			4 :
			4,
		ProblemType::KEYS_ONLY ?		// DOWNSWEEP_LOG_RAKING_THREADS		The number of raking threads (log) for local prefix sum.  Valid range: 5-DOWNSWEEP_LOG_THREADS
			6 :
			6>
{};



/**
 * Sorting Policy
 */
template <typename ProblemType>
struct Policy
{
	enum {
		TUNE_ARCH =
			(__B40C_CUDA_ARCH__ >= 200) ?
				200 :
				(__B40C_CUDA_ARCH__ >= 130) ?
					130 :
					100,
	};

	template <
		int CURRENT_BIT,
		int BITS_REMAINING,
		int CURRENT_PASS>
	struct OpaqueUpsweep :
		typename TunedPassPolicy<
			TUNE_ARCH,
			ProblemType,
			CURRENT_BIT,
			BITS_REMAINING,
			CURRENT_PASS>::Upsweep
	{};

	template <
		int TUNE_ARCH,
		int CURRENT_BIT,
		int BITS_REMAINING,
		int CURRENT_PASS>
	struct PassPolicy :
		TunedPassPolicy<
			TUNE_ARCH,
			ProblemType,
			CURRENT_BIT,
			BITS_REMAINING,
			CURRENT_PASS>
	{};
};


}// namespace radix_sort
}// namespace b40c

