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

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Dispatch policy
 ******************************************************************************/

template <
	int 		_RADIX_BITS,
	int 		_LOG_SCHEDULE_GRANULARITY,
	bool 		_UNIFORM_SMEM_ALLOCATION,
	bool 		_UNIFORM_GRID_SIZE>
struct DispatchPolicy
{
	enum {
		RADIX_BITS					= _RADIX_BITS,
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
	};
};


/******************************************************************************
 * Pass policy
 ******************************************************************************/

/**
 * Pass policy
 */
template <
	typename 	_UpsweepPolicy,
	typename 	_SpinePolicy,
	typename 	_DownsweepPolicy,
	typename 	_DispatchPolicy>
struct PassPolicy
{
	typedef _UpsweepPolicy			UpsweepPolicy;
	typedef _SpinePolicy 			SpinePolicy;
	typedef _DownsweepPolicy 		DownsweepPolicy;
	typedef _DispatchPolicy 		DispatchPolicy;
};



/******************************************************************************
 * Pass dispatch
 ******************************************************************************/

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/kernel_props.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/error_utils.cuh>

#include <b40c/radix_sort/upsweep/kernel_policy.cuh>
#include <b40c/radix_sort/upsweep/kernel.cuh>

#include <b40c/radix_sort/spine/kernel_policy.cuh>
#include <b40c/radix_sort/spine/kernel.cuh>
#include <b40c/radix_sort/spine/tex_ref.cuh>

#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel.cuh>
#include <b40c/radix_sort/downsweep/tex_ref.cuh>

/**
 * Pass instance
 */
template <
	typename 	KeyType,
	typename 	ValueType,
	typename 	SizeT
	int 		BITS_REMAINING,
	int 		CURRENT_BIT,
	int 		CURRENT_PASS>
struct PassInstance
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	// Kernel function types
	typedef void (*UpsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(SizeT*, SizeT*, int);
	typedef void (*DownsweepKernelPtr)(int*, SizeT*, KeyType*, KeyType*, ValueType*, ValueType*, util::CtaWorkDistribution<SizeT>);

	// Texture binding function types
	typedef cudaError_t (*BindDownsweepTexture)(void *, void *, size_t, void *, void *, size_t);
	typedef cudaError_t (*BindSpineTexture)(void *, size_t);


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	Spine				&spine;
	DoubleBuffer		&problem_storage;
	SizeT				num_elements;
	int			 		max_grid_size;
	int 				ptx_arch;
	int 				sm_arch;
	bool				debug;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	PassInstance(
		Spine				&spine,
		DoubleBuffer		&problem_storage,
		SizeT				num_elements,
		int			 		max_grid_size,
		int 				ptx_arch,
		int 				sm_arch,
		bool				debug) :
			spine(spine),
			problem_storage(problem_storage),
			num_elements(num_elements),
			max_grid_size(max_grid_size),
			ptx_arch(ptx_arch),
			sm_arch(sm_arch),
			debug(debug)
	{}


	/**
	 * Dispatch
	 */
	cudaError_t Dispatch(
		int 									radix_bits,
		const KernelProps<UpsweepKernelPtr> 	&upsweep_props,
		const KernelProps<SpineKernelPtr> 		&spine_props,
		const KernelProps<DownsweepKernelPtr> 	&downsweep_props,
		BindDownsweepTexture 					bind_downsweep_texture_ptr,
		BindValueTextures 						bind_value_textures_ptr,
		BindSpineTexture 						bind_spine_texture_ptr,
		int 									log_schedule_granularity,
		int										spine_tile_elements,
		bool									smem_8byte_banks,
		bool									unform_grid_size,
		bool									uniform_smem_allocation)
	{
		cudaError_t retval = cudaSuccess;
		do {

			if (debug) {
				printf("Current bit %d, Pass %d, Radix bits %d:\n",
					CURRENT_BIT,
					CURRENT_PASS,
					radix_bits);

				printf("Upsweep occupancy %d, downsweep occupancy %d\n",
					upsweep_props.max_cta_occupancy,
					downsweep_props.max_cta_occupancy);
			}

			// Compute sweep grid size
			int schedule_granularity = 1 << log_schedule_granularity;
			int sweep_grid_size = downsweep_props.OversubscribedGridSize(
				schedule_granularity,
				num_elements,
				max_grid_size);

			// Compute spine elements (rounded up to nearest tile size)
			SizeT spine_elements = CUB_ROUND_UP_NEAREST(
				sweep_grid_size << radix_bits,
				spine_tile_elements);

			// Make sure our spine is big enough
			if (retval = spine.Setup<SizeT>(spine_elements)) break;

			// Obtain a CTA work distribution
			util::CtaWorkDistribution<SizeT> work;
			work.Init(detail.num_elements, sweep_grid_size, log_schedule_granularity);
			if (debug) {
				work.Print();
			}

			// Bind downsweep textures
			if (bind_downsweep_texture_ptr != NULL) {
				if (retval = bind_downsweep_texture_ptr(
					problem_storage.d_keys[problem_storage.selector],
					problem_storage.d_keys[problem_storage.selector ^ 1],
					num_elements * sizeof(KeyType),
					problem_storage.d_values[problem_storage.selector],
					problem_storage.d_values[problem_storage.selector ^ 1],
					num_elements * sizeof(ValueType))) break;
			}

			// Bind spine textures
			if (bind_spine_texture_ptr != NULL) {
				if (retval = bind_spine_texture_ptr(
					spine(),
					spine_elements * sizeof(SizeT))) break;
			}

			// Operational details
			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{sweep_grid_size, 1, sweep_grid_size};

			// Grid size tuning
			if (unform_grid_size) {
				// Make sure that all kernels launch the same number of CTAs
				grid_size[1] = grid_size[0];
			}

			// Smem allocation tuning
			if (uniform_smem_allocation) {

				// Make sure all kernels have the same overall smem allocation
				int max_static_smem = CUB_MAX(
					upsweep_props.kernel_attrs.sharedSizeBytes,
					CUB_MAX(
						spine_props.kernel_attrs.sharedSizeBytes,
						downsweep_props.kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - pine_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_props.kernel_attrs.sharedSizeBytes;

			} else {

				// Compute smem padding for upsweep to make upsweep occupancy a multiple of downsweep occupancy
				dynamic_smem[0] = upsweep_props.SmemPadding(downsweep_props.max_cta_occupancy);
			}

			// Upsweep reduction into spine
			upsweep_props.kenrel_ptr<<<grid_size[0], upsweep_props.threads, dynamic_smem[0]>>>(
				(SizeT*) spine(),
				(KeyType *) problem_storage.d_keys[problem_storage.selector],
				(KeyType *) problem_storage.d_keys[problem_storage.selector ^ 1],
				work);

			if (debug && (retval = util::B40CPerror(cudaThreadSynchronize(), "Upsweep kernel failed ", __FILE__, __LINE__, debug))) break;

			// Spine scan
			spine_props.kenrel_ptr<<<grid_size[1], spine_props.threads, dynamic_smem[1]>>>(
				(SizeT*) spine(),
				(SizeT*) spine(),
				spine_elements);

			if (debug && (retval = util::B40CPerror(cudaThreadSynchronize(), "Spine kernel failed ", __FILE__, __LINE__, debug))) break;

			// Set shared mem bank mode
			enum cudaSharedMemConfig old_config;
			cudaDeviceGetSharedMemConfig(&old_config);
			cudaDeviceSetSharedMemConfig(smem_8byte_banks ?
				cudaSharedMemBankSizeEightByte :		// 64-bit bank mode
				cudaSharedMemBankSizeFourByte);			// 32-bit bank mode

			// Downsweep scan from spine
			downsweep_props.kenrel_ptr<<<grid_size[2], downsweep_props.threads, dynamic_smem[2]>>>(
				d_selectors,
				(SizeT *) spine(),
				(KeyType *) problem_storage.d_keys[problem_storage.selector],
				(KeyType *) problem_storage.d_keys[problem_storage.selector ^ 1],
				problem_storage.d_values[problem_storage.selector],
				problem_storage.d_values[problem_storage.selector ^ 1],
				work);

			if (debug && (retval = util::B40CPerror(cudaThreadSynchronize(), "Downsweep kernel failed ", __FILE__, __LINE__, debug))) break;

			// Restore smem bank mode
			cudaDeviceSetSharedMemConfig(old_config);

		} while(0);

		return retval;
	}


	/**
	 * Dispatch
	 */
	template <
		typename HostPassPolicy,
		typename DevicePassPolicy>
	cudaError_t Dispatch()
	{
		typedef typename HostPassPolicy::UpsweepPolicy 		UpsweepPolicy;
		typedef typename HostPassPolicy::SpinePolicy 		SpinePolicy;
		typedef typename HostPassPolicy::DownsweepPolicy 	DownsweepPolicy;
		typedef typename HostPassPolicy::DispatchPolicy	 	DispatchPolicy;

		// Upsweep kernel properties
		KernelProps<UpsweepKernelPtr> upsweep_props(
			upsweep::Kernel<typename DevicePassPolicy::UpsweepPolicy>,
			UpsweepPolicy::THREADS,
			sm_arch);

		// Spine kernel properties
		KernelProps<SpineKernelPtr> spine_props(
			spine::Kernel<typename DevicePassPolicy::SpinePolicy>,
			SpinePolicy::THREADS,
			sm_arch);

		// Downsweep kernel properties
		KernelProps<DownsweepKernelPtr> downsweep_props(
			downsweep::Kernel<typename DevicePassPolicy::DownsweepPolicy>,
			DownsweepPolicy::THREADS,
			sm_arch);

		// Whether to use 8-byte bank mode
		bool smem_8byte_banks = DownsweepPolicy::SMEM_8BYTE_BANKS;

		// Schedule granularity
		int log_schedule_granularity = CUB_MAX(
			UpsweepPolicy::LOG_TILE_ELEMENTS,
			DownsweepPolicy::LOG_TILE_ELEMENTS);

		// Radix bits
		int radix_bits = DispatchPolicy::RADIX_BITS;

		// Spine tile elements
		int spine_tile_elements = SpinePolicy::TILE_ELEMENTS;

		// Texture binding functions
		BindKeyTextures bind_downsweep_texture_ptr = downsweep::DownsweepTex<
			KeyType,
			ValueType,
			1 << DownsweepPolicy::LOG_THREAD_ELEMENTS>::BindTextures;

		BindSpineTexture bind_spine_texture_ptr = spine::SpineTex<SizeT>::BindTexture;

		return Dispatch(
			radix_bits,
			upsweep_props,
			spine_props,
			downsweep_props,
			bind_downsweep_texture_ptr,
			bind_spine_texture_ptr,
			log_schedule_granularity,
			spine_tile_elements,
			smem_8byte_banks);
	}


	/**
	 * Dispatch
	 */
	template <PassPolicy>
	cudaError_t Dispatch()
	{
		return Dispatch(PassPolicy, PassPolicy);
	}


	//---------------------------------------------------------------------
	// Preconfigured pass dispatch
	//---------------------------------------------------------------------

	/**
	 * Specialized pass policies
	 */
	template <TUNE_ARCH>
	struct TunedPassPolicy;


	/**
	 * SM20
	 */
	template <>
	struct TunedPassPolicy<200>
	{
		enum {
			RADIX_BITS 		= CUB_MIN(BITS_REMAINING, 5),
			KEYS_ONLY 		= util::Equals<ValueType, util::NullType>::VALUE,
			EARLY_EXIT 		= false,
		};

		// Upsweep kernel policy
		typedef upsweep::KernelPolicy<
			RADIX_BITS,						// RADIX_BITS
			CURRENT_BIT,					// CURRENT_BIT
			CURRENT_PASS,					// CURRENT_PASS
			8,								// MIN_CTA_OCCUPANCY	The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
			7,								// LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10
			2,								// LOG_LOAD_VEC_SIZE	The vector-load size (log) for each load (log).  Valid range: 0-2
			1,								// LOG_LOADS_PER_TILE	The number of loads (log) per tile.  Valid range: 0-2
			b40c::util::io::ld::NONE,		// READ_MODIFIER		Load cache-modifier.  Valid values: NONE, ca, cg, cs
			b40c::util::io::st::NONE,		// WRITE_MODIFIER		Store cache-modifier.  Valid values: NONE, wb, cg, cs
			EARLY_EXIT>						// EARLY_EXIT			Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
				UpsweepPolicy;

		// Spine-scan kernel policy
		typedef spine::KernelPolicy<
			8,								// LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10
			2,								// LOG_LOAD_VEC_SIZE	The vector-load size (log) for each load (log).  Valid range: 0-2
			2,								// LOG_LOADS_PER_TILE	The number of loads (log) per tile.  Valid range: 0-2
			b40c::util::io::ld::NONE,		// READ_MODIFIER		Load cache-modifier.  Valid values: NONE, ca, cg, cs
			b40c::util::io::st::NONE>		// WRITE_MODIFIER		Store cache-modifier.  Valid values: NONE, wb, cg, cs
				SpinePolicy;

		// Downsweep kernel policy
		typedef downsweep::KernelPolicy<
			RADIX_BITS,						// RADIX_BITS
			CURRENT_BIT,					// CURRENT_BIT
			CURRENT_PASS,					// CURRENT_PASS
			KEYS_ONLY ? 4 : 2,				// MIN_CTA_OCCUPANCY		The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
			KEYS_ONLY ? 7 : 8,				// LOG_THREADS				The number of threads (log) to launch per CTA.
			KEYS_ONLY ? 4 : 4,				// LOG_ELEMENTS_PER_TILE	The number of keys (log) per thread
			b40c::util::io::ld::NONE,		// READ_MODIFIER			Load cache-modifier.  Valid values: NONE, ca, cg, cs
			b40c::util::io::st::NONE,		// WRITE_MODIFIER			Store cache-modifier.  Valid values: NONE, wb, cg, cs
			downsweep::SCATTER_TWO_PHASE,	// SCATTER_STRATEGY			Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
			false,							// SMEM_8BYTE_BANKS
			EARLY_EXIT>						// EARLY_EXIT				Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
				DownsweepPolicy;

		// Dispatch policy
		typedef radix_sort::DispatchPolicy <
			RADIX_BITS,							// RADIX_BITS
			false, 								// UNIFORM_SMEM_ALLOCATION
			true, 								// UNIFORM_GRID_SIZE
			true>  								// OVERSUBSCRIBED_GRID_SIZE
				DispatchPolicy;
	};


	/**
	 * Opaque pass policy
	 */
	template <int PTX_ARCH>
	struct OpaquePassPolicy
	{
		struct UpsweepPolicy : 		TunedPassPolicy<PTX_ARCH>::UpsweepPolicy {};
		struct SpinePolicy : 		TunedPassPolicy<PTX_ARCH>::SpinePolicy {};
		struct DownsweepPolicy : 	TunedPassPolicy<PTX_ARCH>::DownsweepPolicy {};
		struct DispatchPolicy : 	TunedPassPolicy<PTX_ARCH>::DispatchPolicy {};
	};


	/**
	 * The appropriate tuning arch-id from the arch-id targeted by the
	 * active compiler pass.
	 */
	enum {
		TUNE_ARCH =
			(__B40C_CUDA_ARCH__ >= 200) ?
				200 :
				(__B40C_CUDA_ARCH__ >= 130) ?
					130 :
					100,
	};


	/**
	 * Dispatch
	 */
	cudaError_t Dispatch()
	{
		if (ptx_arch >= 200) {

			return Disaptch<TunedPassPolicy<200>, OpaquePassPolicy<TUNE_ARCH> >();

		} else if (ptx_arch >= 130) {

			return Disaptch<TunedPassPolicy<130>, OpaquePassPolicy<TUNE_ARCH> >();

		} else {

			return Disaptch<TunedPassPolicy<100>, OpaquePassPolicy<TUNE_ARCH> >();
		}
	}


};


}// namespace radix_sort
}// namespace b40c

