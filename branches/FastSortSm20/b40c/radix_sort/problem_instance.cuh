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
 * Radix sorting enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/spine.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/kernel_props.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/radix_sort/sort_utils.cuh>
#include <b40c/radix_sort/upsweep/kernel_policy.cuh>
#include <b40c/radix_sort/upsweep/kernel.cuh>

#include <b40c/radix_sort/spine/kernel_policy.cuh>
#include <b40c/radix_sort/spine/kernel.cuh>
#include <b40c/radix_sort/spine/tex_ref.cuh>

#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel.cuh>
#include <b40c/radix_sort/downsweep/tex_ref.cuh>

namespace b40c {
namespace radix_sort {



/******************************************************************************
 * Problem instance
 ******************************************************************************/

/**
 * Problem instance
 */
template <typename DoubleBuffer, typename _SizeT>
struct ProblemInstance
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	typedef typename DoubleBuffer::KeyType 					KeyType;
	typedef typename DoubleBuffer::ValueType 				ValueType;
	typedef _SizeT 											SizeT;
	typedef typename KeyTraits<KeyType>::IngressOp 			IngressOp;
	typedef typename KeyTraits<KeyType>::EgressOp 			EgressOp;
	typedef typename KeyTraits<KeyType>::ConvertedKeyType 	ConvertedKeyType;

	/**
	 * Upsweep kernel properties
	 */
	struct UpsweepKernelProps : util::KernelProps
	{
		// Upsweep kernel function type
		typedef void (*KernelFunc)(SizeT*, ConvertedKeyType*, ConvertedKeyType*, IngressOp, util::CtaWorkDistribution<SizeT>);

		// Fields
		KernelFunc 	kernel_ptr;
		int 		log_tile_elements;
		bool 		smem_8byte_banks;

		/**
		 * Initializer
		 */
		template <
			typename KernelPolicy,
			typename OpaquePolicy = KernelPolicy>
		cudaError_t Init(int sm_arch, int sm_count)
		{
			// Initialize fields
			kernel_ptr 			= upsweep::Kernel<OpaquePolicy>;
			log_tile_elements 	= KernelPolicy::LOG_TILE_ELEMENTS;
			smem_8byte_banks 	= KernelPolicy::SMEM_8BYTE_BANKS;

			// Initialize super class
			return util::KernelProps::Init(
				kernel_ptr,
				KernelPolicy::THREADS,
				sm_arch,
				sm_count);
		}
	};


	/**
	 * Spine kernel properties
	 */
	struct SpineKernelProps : util::KernelProps
	{
		// Spine kernel function type
		typedef void (*KernelFunc)(SizeT*, SizeT*, int);

		// Spine texture binding function type
		typedef cudaError_t (*BindTexFunc)(void *, size_t);

		// Fields
		KernelFunc 		kernel_ptr;
		BindTexFunc		bind_tex_func;
		int 			log_tile_elements;

		/**
		 * Initializer
		 */
		template <
			typename KernelPolicy,
			typename OpaquePolicy = KernelPolicy>
		cudaError_t Init(int sm_arch, int sm_count)
		{
			// Initialize fields
			kernel_ptr 			= upsweep::Kernel<OpaquePolicy>;
			bind_tex_func 		= spine::TexSpine<SizeT>::BindTexture;
			log_tile_elements 	= KernelPolicy::LOG_TILE_ELEMENTS;

			// Initialize super class
			return util::KernelProps::Init(
				kernel_ptr,
				KernelPolicy::THREADS,
				sm_arch,
				sm_count);
		}

		// Bind related textures
		cudaErrorT BindTexture(SizeT *spine, int spine_elements)
		{
			return bind_tex_func(spine, sizeof(SizeT) * spine_elements);
		}
	};


	/**
	 * Downsweep kernel props
	 */
	struct DownsweepKernelProps : util::KernelProps
	{
		// Downsweep kernel function type
		typedef void (*KernelFunc)(SizeT*, ConvertedKeyType*, ConvertedKeyType*, ValueType*, ValueType*, IngressOp, EgressOp, util::CtaWorkDistribution<SizeT>);

		// Downsweep texture binding function type
		typedef cudaError_t (*BindTexFunc)(void *, void *, size_t);

		// Fields
		KernelFunc 		kernel_ptr;
		BindTexFunc		keys_tex_func;
		BindTexFunc		values_tex_func;
		int 			log_tile_elements;
		bool 			smem_8byte_banks;

		/**
		 * Initializer
		 */
		template <
			typename KernelPolicy,
			typename OpaquePolicy = KernelPolicy>
		cudaError_t Init(int sm_arch, int sm_count)
		{
			const int THREAD_ELEMENTS = 1 << KernelPolicy::LOG_THREAD_ELEMENTS;

			// Wrapper of downsweep texture types
			typedef downsweep::Textures<KeyType, ValueType, THREAD_ELEMENTS> DownsweepTextures;

			// Key texture type
			typedef typename DownsweepTextures::KeyTexType KeyTexType;

			// Value texture type
			typedef typename DownsweepTextures::ValueTexType ValueTexType;

			// Initialize fields
			kernel_ptr 			= downsweep::Kernel<OpaquePolicy>;
			keys_tex_func 		= downsweep::TexKeys<KeyTexType>::BindTexture;
			values_tex_func 	= downsweep::TexValues<ValueTexType>::BindTexture;
			log_tile_elements 	= KernelPolicy::LOG_TILE_ELEMENTS;
			smem_8byte_banks	= KernelPolicy::SMEM_8BYTE_BANKS;

			// Initialize super class
			return util::KernelProps::Init(
				kernel_ptr,
				KernelPolicy::THREADS,
				sm_arch,
				sm_count);
		}

		// Bind related textures
		cudaErrorT BindTexture(
			KeyType *d_keys0,
			KeyType *d_keys1,
			ValueType *d_values0,
			ValueType *d_values1,
			SizeT num_elements)
		{
			cudaError error = cudaSucces;
			do {
				// Bind key texture
				error = keys_tex_func(d_keys0, d_keys1, sizeof(KeyType) * num_elements);
				if (error) break;

				// Bind value texture
				error = values_tex_func(d_values0, d_values1, sizeof(ValueType) * num_elements);
				if (error) break;

			} while (0);

			return error;
		}
	};


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	DoubleBuffer		&storage;
	SizeT				num_elements;
	IngressOp			ingress_op;
	EgressOp			egress_op;

	util::Spine			&spine;
	cudaStream_t		stream;
	int			 		max_grid_size;
	bool				debug;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	ProblemInstance(
		DoubleBuffer	&storage,
		SizeT			num_elements,
		util::Spine		&spine,
		int			 	max_grid_size,
		bool			debug) :
			storage(storage),
			num_elements(num_elements),
			spine(spine),
			max_grid_size(max_grid_size),
			debug(debug)
	{}


	/**
	 * DispatchPass
	 */
	cudaError_t DispatchPass(
		int 							radix_bits,
		const UpsweepKernelProps 		&upsweep_props,
		const SpineKernelProps			&spine_props,
		const DownsweepKernelProps		&downsweep_props,
		bool							unform_grid_size,
		bool							uniform_smem_allocation)
	{
		cudaError_t error = cudaSuccess;

		do {
			// Compute sweep grid size
			int log_schedule_granularity = CUB_MAX(
				upsweep_props.log_tile_elements,
				downsweep_props.log_tile_elements);
			int sweep_grid_size = downsweep_props.OversubscribedGridSize(
				1 << log_schedule_granularity,
				num_elements,
				max_grid_size);

			// Compute spine elements (rounded up to nearest tile size)
			SizeT spine_elements = CUB_ROUND_UP_NEAREST(
				sweep_grid_size << radix_bits,
				spine_props.tile_elements);

			// Make sure our spine is big enough
			error = spine.Setup(sizeof(SizeT) * spine_elements);
			if (error) break;

			// Obtain a CTA work distribution
			util::CtaWorkDistribution<SizeT> work(
				num_elements,
				sweep_grid_size,
				log_schedule_granularity);

			// Bind spine textures
			error = spine_props.BindTexture(spine(), spine_elements);
			if (error) break;

			// Bind downsweep textures
			error = downsweep_props.BindTexture(
				storage.d_keys[storage.selector],
				storage.d_keys[storage.selector ^ 1],
				storage.d_values[storage.selector],
				storage.d_values[storage.selector ^ 1],
				num_elements);
			if (error) break;

			// Grid size tuning
			int grid_size[3] = {sweep_grid_size, 1, sweep_grid_size};
			if (unform_grid_size)
			{
				// Make sure that all kernels launch the same number of CTAs
				grid_size[1] = grid_size[0];
			}

			// Smem allocation tuning
			int dynamic_smem[3] = {0, 0, 0};
			if (uniform_smem_allocation)
			{
				// Make sure all kernels have the same overall smem allocation
				int max_static_smem = CUB_MAX(
					upsweep_props.kernel_attrs.sharedSizeBytes,
					CUB_MAX(
						spine_props.kernel_attrs.sharedSizeBytes,
						downsweep_props.kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_props.kernel_attrs.sharedSizeBytes;
			}
			else
			{
				// Compute smem padding for upsweep to make upsweep occupancy a multiple of downsweep occupancy
				dynamic_smem[0] = upsweep_props.SmemPadding(downsweep_props.max_cta_occupancy);
			}

			// Print debug info
			if (debug)
			{
				work.Print();
				printf(
					"Upsweep:   tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Spine:     tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Downsweep: tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n",
					upsweep_props.tile_elements, upsweep_props.max_cta_occupancy, grid_size[0], upsweep_props.threads, dynamic_smem[0],
					spine_props.tile_elements, spine_props.max_cta_occupancy, grid_size[1], spine_props.threads, dynamic_smem[1],
					downsweep_props.tile_elements, downsweep_props.max_cta_occupancy, grid_size[2], downsweep_props.threads, dynamic_smem[2]);
				fflush(stdout);
			}

			//
			// Upsweep
			//

			// Set shared mem bank mode
			enum cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			cudaDeviceSetSharedMemConfig(smem_8byte_banks ?
				cudaSharedMemBankSizeEightByte :		// 64-bit bank mode
				cudaSharedMemBankSizeFourByte);			// 32-bit bank mode

			// Upsweep reduction into spine
			upsweep_props.kernel_func<<<grid_size[0], upsweep_props.threads, dynamic_smem[0]>>>(
				(SizeT*) spine(),
				(ConvertedKeyType *) storage.d_keys[storage.selector],
				(ConvertedKeyType *) storage.d_keys[storage.selector ^ 1],
				ingress_op,
				work);

			// Restore smem bank mode
			cudaDeviceSetSharedMemConfig(old_sm_config);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Upsweep kernel failed ", __FILE__, __LINE__)) break;
			}

			//
			// Spine
			//

			// Spine scan
			spine_props.kernel_func<<<grid_size[1], spine_props.threads, dynamic_smem[1]>>>(
				(SizeT*) spine(),
				(SizeT*) spine(),
				spine_elements);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Spine kernel failed ", __FILE__, __LINE__)) break;
			}

			//
			// Downsweep
			//

			// Set shared mem bank mode
			enum cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			cudaDeviceSetSharedMemConfig(smem_8byte_banks ?
				cudaSharedMemBankSizeEightByte :		// 64-bit bank mode
				cudaSharedMemBankSizeFourByte);			// 32-bit bank mode

			// Downsweep scan from spine
			downsweep_props.kernel_func<<<grid_size[2], downsweep_props.threads, dynamic_smem[2]>>>(
				(SizeT *) spine(),
				(ConvertedKeyType *) storage.d_keys[storage.selector],
				(ConvertedKeyType *) storage.d_keys[storage.selector ^ 1],
				storage.d_values[storage.selector],
				storage.d_values[storage.selector ^ 1],
				ingress_op,
				egress_op,
				work);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Downsweep kernel failed ", __FILE__, __LINE__)) break;
			}

			// Restore smem bank mode
			cudaDeviceSetSharedMemConfig(old_sm_config);

		} while(0);

		return error;
	}
};




} // namespace radix_sort
} // namespace b40c

