/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * Radix sorting problem instance
 ******************************************************************************/

#pragma once

#include "../cub/cub.cuh"

#include "ns_wrapper.cuh"
#include "radix_sort/tuned_policy.cuh"
#include "radix_sort/sort_utils.cuh"
#include "radix_sort/kernel_upsweep_pass.cuh"
#include "radix_sort/kernel_scan_pass.cuh"
#include "radix_sort/kernel_downsweep_pass.cuh"
#include "radix_sort/kernel_single_tile.cuh"
#include "radix_sort/kernel_hybrid_pass.cuh"

BACK40_NS_PREFIX
namespace back40 {


/******************************************************************************
 * Problem instance
 ******************************************************************************/

/**
 * Problem instance
 */
template <
	typename Allocator,
	typename KeyType,
	typename ValueType,
	typename SizeT>
struct GpuRadixSort
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	/**
	 * Tuned pass policy whose type signature does not reflect the tuned
	 * SM architecture.
	 */
	template <radix_sort::ProblemSize PROBLEM_SIZE>
	struct OpaqueTunedPolicy
	{
		// The appropriate tuning arch-id from the arch-id targeted by the
		// active compiler pass.
		enum
		{
/*
			COMPILER_TUNE_ARCH 		= (__CUB_CUDA_ARCH__ >= 200) ?
										200 :
										(__CUB_CUDA_ARCH__ >= 130) ?
											130 :
											100
*/
			COMPILER_TUNE_ARCH = 200
		};

		// Tuned pass policy
		typedef radix_sort::TunedPolicy<
			COMPILER_TUNE_ARCH,
			KeyType,
			ValueType,
			SizeT,
			PROBLEM_SIZE> TunedPolicy;

		struct DispatchPolicyT 				: TunedPolicy::DispatchPolicyT {};
		struct CtaUpsweepPassPolicyT 		: TunedPolicy::CtaUpsweepPassPolicyT {};
		struct CtaScanPassPolicyT 			: TunedPolicy::CtaScanPassPolicyT {};
		struct CtaDownsweepPassPolicyT 		: TunedPolicy::CtaDownsweepPassPolicyT {};
		struct CtaSingleTilePolicyT 		: TunedPolicy::CtaSingleTilePolicyT {};
		struct CtaHybridPassPolicyT 		: TunedPolicy::CtaHybridPassPolicyT {};
	};


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	cub::CudaProps					*cuda_props;
	Allocator						*allocator;

	// Kernel properties
	radix_sort::UpsweepKernelProps<KeyType, SizeT> 					upsweep_props;
	radix_sort::SpineKernelProps<SizeT> 							spine_props;
	radix_sort::DownsweepKernelProps<KeyType, ValueType, SizeT> 	downsweep_props;
	radix_sort::SingleTileKernelProps<KeyType, ValueType, SizeT> 	single_tile_props;
	radix_sort::HybridKernelProps<KeyType, ValueType, SizeT> 		hybrid_props;

	bool							uniform_grid_size;
	radix_sort::DynamicSmemConfig	dynamic_smem_config;

	int 							selector;
	KeyType							*d_keys[2];
	ValueType						*d_values[2];
	SizeT							*d_spine;
	radix_sort::BinDescriptor		*d_bins[2];

	size_t							spine_bytes;

	SizeT							num_elements;
	int 							low_bit;
	int								num_bits;
	cudaStream_t					stream;
	int			 					max_grid_size;
	bool							debug;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	GpuRadixSort(
		cub::CudaProps		*cuda_props,
		Allocator			*allocator,
		bool 				debug = false) :
			cuda_props(cuda_props),
			allocator(allocator),
			selector(0),
			d_spine(NULL),
			spine_bytes(0),
			debug(debug)
	{
		this->d_keys[0] 		= NULL;
		this->d_keys[1] 		= NULL;
		this->d_values[0] 		= NULL;
		this->d_values[1] 		= NULL;
		this->d_bins[0] 		= NULL;
		this->d_bins[1]			= NULL;
	}


	/**
	 * Destructor
	 */
	virtual ~GpuRadixSort()
	{
		if (allocator)
		{
			if (d_keys[1]) allocator->Deallocate(d_keys[1]);
			if (d_values[1]) allocator->Deallocate(d_values[1]);
			if (d_bins[0]) allocator->Deallocate(d_bins[0]);
			if (d_bins[1]) allocator->Deallocate(d_bins[1]);
			if (d_spine) allocator->Deallocate(d_spine);
			spine_bytes = 0;
		}
	}

	/**
	 * Configure with default autotuned kernel properties
	 */
	template <
		int TUNE_ARCH,
		radix_sort::ProblemSize PROBLEM_SIZE>
	cudaError_t Configure()
	{
		// Tuned policy
		typedef radix_sort::TunedPolicy<
			TUNE_ARCH,
			KeyType,
			ValueType,
			SizeT,
			PROBLEM_SIZE> TunedPolicy;

		// Opaque tuned policy
		typedef OpaqueTunedPolicy<PROBLEM_SIZE> OpaqueTunedPolicy;

		// Print debug info
		if (debug)
		{
			printf("Tuned arch(%d), SM arch(%d)\n",
				TUNE_ARCH,
				cuda_props->sm_version);
			fflush(stdout);
		}

		cudaError_t error = cudaSuccess;
		do
		{
			// Initialize upsweep kernel props
			error = upsweep_props.template Init<
				typename TunedPolicy::CtaUpsweepPassPolicyT,
				typename OpaqueTunedPolicy::CtaUpsweepPassPolicyT,
				TunedPolicy::DispatchPolicyT::UPSWEEP_MIN_CTA_OCCUPANCY>(*cuda_props);
			if (CubDebug(error)) break;

			// Initialize spine kernel props
			error = spine_props.template Init<
				typename TunedPolicy::CtaScanPassPolicyT,
				typename OpaqueTunedPolicy::CtaScanPassPolicyT>(*cuda_props);
			if (CubDebug(error)) break;

			// Initialize downsweep kernel props
			error = downsweep_props.template Init<
				typename TunedPolicy::CtaDownsweepPassPolicyT,
				typename OpaqueTunedPolicy::CtaDownsweepPassPolicyT,
				TunedPolicy::DispatchPolicyT::DOWNSWEEP_MIN_CTA_OCCUPANCY>(*cuda_props);
			if (CubDebug(error)) break;

			// Initialize single-tile kernel props
			error = single_tile_props.template Init<
				typename TunedPolicy::CtaSingleTilePolicyT,
				typename OpaqueTunedPolicy::CtaSingleTilePolicyT>(*cuda_props);
			if (CubDebug(error)) break;

			// Initialize hybrid kernel props
			error = hybrid_props.template Init<
				typename TunedPolicy::CtaHybridPassPolicyT,
				typename OpaqueTunedPolicy::CtaHybridPassPolicyT,
				TunedPolicy::DispatchPolicyT::HYBRID_MIN_CTA_OCCUPANCY>(*cuda_props);
			if (CubDebug(error)) break;

			uniform_grid_size 		= TunedPolicy::DispatchPolicyT::UNIFORM_GRID_SIZE;
			dynamic_smem_config 	= TunedPolicy::DispatchPolicyT::DYNAMIC_SMEM_CONFIG;

		} while (0);

		return error;
	}


	/**
	 * Configure with specified kernel properties
	 */
	cudaError_t Configure(
		radix_sort::UpsweepKernelProps<KeyType, SizeT> 					upsweep_props,
		radix_sort::SpineKernelProps<SizeT> 							spine_props,
		radix_sort::DownsweepKernelProps<KeyType, ValueType, SizeT> 	downsweep_props,
		radix_sort::SingleTileKernelProps<KeyType, ValueType, SizeT> 	single_tile_props,
		radix_sort::HybridKernelProps<KeyType, ValueType, SizeT> 		hybrid_props,
		bool															uniform_grid_size,
		radix_sort::DynamicSmemConfig									dynamic_smem_config)
	{
		this->upsweep_props 		= upsweep_props;
		this->spine_props 			= spine_props;
		this->downsweep_props 		= downsweep_props;
		this->single_tile_props 	= single_tile_props;
		this->hybrid_props 			= hybrid_props;
		this->uniform_grid_size		= uniform_grid_size;
		this->dynamic_smem_config	= dynamic_smem_config;

		return cudaSuccess;
	}


	/**
	 * Resize spine
	 */
	cudaError_t ResizeSpine(SizeT spine_elements)
	{
		cudaError_t error = cudaSuccess;
		do
		{
			size_t spine_bytes_needed = sizeof(SizeT) * spine_elements;
			if (spine_bytes_needed > spine_bytes)
			{
				if (d_spine)
				{
					// Deallocate
					error = allocator->Deallocate(d_spine);
					if (CubDebug(error)) break;
				}
				// Allocate
				error = allocator->Allocate((void**) &d_spine, spine_bytes_needed);
				if (CubDebug(error)) break;
				spine_bytes = spine_bytes_needed;
			}

		} while (0);

		return error;
	}


	/**
	 * Dispatch global partitioning pass
	 */
	cudaError_t DispatchGlobalPass()
	{
		cudaError_t error = cudaSuccess;

		int radix_digits = 1 << upsweep_props.radix_bits;
		do {
			// Current bit
			int current_bit = low_bit + num_bits - upsweep_props.radix_bits;

			// Compute sweep grid size
			int schedule_granularity = CUB_MAX(upsweep_props.tile_items, downsweep_props.tile_items);
			int sweep_grid_size = downsweep_props.OversubscribedGridSize(
				schedule_granularity,
				num_elements,
				max_grid_size);

			// Compute spine elements (rounded up to nearest tile size)
			SizeT spine_elements = CUB_ROUND_UP_NEAREST(
				sweep_grid_size * radix_digits,						// Each CTA produces a partial for every radix digit
				spine_props.tile_items);							// Number of partials per tile

			// Allocate spine
			error = ResizeSpine(spine_elements);
			if (CubDebug(error)) break;

			// Obtain a CTA work distribution
			cub::CtaWorkDistribution<SizeT> work(
				num_elements,
				sweep_grid_size,
				schedule_granularity);

			// Configure grid size
			int grid_size[3] = {sweep_grid_size, 1, sweep_grid_size};
			if (uniform_grid_size)
			{
				// Make sure that all kernels launch the same number of CTAs
				grid_size[1] = grid_size[0];
			}

			// Configure dynamic smem allocation
			int dynamic_smem[3] = {0, 0, 0};
			if (dynamic_smem_config == radix_sort::DYNAMIC_SMEM_UNIFORM)
			{
				// Pad with dynamic smem so all kernels get the same total smem allocation
				int max_static_smem = CUB_MAX(
					upsweep_props.cta_allocated_smem,
					CUB_MAX(
						spine_props.cta_allocated_smem,
						downsweep_props.cta_allocated_smem));

				dynamic_smem[0] = max_static_smem - upsweep_props.cta_allocated_smem;
				dynamic_smem[1] = max_static_smem - spine_props.cta_allocated_smem;
				dynamic_smem[2] = max_static_smem - downsweep_props.cta_allocated_smem;
			}
			else if (dynamic_smem_config == radix_sort::DYNAMIC_SMEM_LCM)
			{
				// Pad upsweep/downsweep with dynamic smem so kernel occupancy a multiple of the lowest occupancy
				int min_occupancy = CUB_MIN(upsweep_props.max_cta_occupancy, downsweep_props.max_cta_occupancy);
				dynamic_smem[0] = upsweep_props.DynamicSmemPadding(min_occupancy);
				dynamic_smem[2] = downsweep_props.DynamicSmemPadding(min_occupancy);
			}

			// Print debug info
			if (debug)
			{
				work.Print();
				printf(
					"Global %d-bit pass at bit(%d):\n"
					"Upsweep:   tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Spine:     tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Downsweep: tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n",
					upsweep_props.radix_bits, current_bit,
					upsweep_props.tile_items, upsweep_props.max_cta_occupancy, grid_size[0], upsweep_props.cta_threads, dynamic_smem[0],
					spine_props.tile_items, spine_props.max_cta_occupancy, grid_size[1], spine_props.cta_threads, dynamic_smem[1],
					downsweep_props.tile_items, downsweep_props.max_cta_occupancy, grid_size[2], downsweep_props.cta_threads, dynamic_smem[2]);
				fflush(stdout);
			}

			// Set upsweep shared mem bank mode
			cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			if (old_sm_config != upsweep_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(upsweep_props.sm_bank_config);

			// Upsweep reduction into spine
			upsweep_props.kernel_func<<<grid_size[0], upsweep_props.cta_threads, dynamic_smem[0], stream>>>(
				d_keys[selector],
				d_spine,
				work,
				current_bit);
			if (debug && (error = CubDebug(cudaThreadSynchronize()))) break;

			// Set spine shared mem bank mode
			if (spine_props.sm_bank_config != upsweep_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(spine_props.sm_bank_config);

			// Spine scan
			spine_props.kernel_func<<<grid_size[1], spine_props.cta_threads, dynamic_smem[1], stream>>>(
				d_spine,
				d_spine,
				spine_elements);
			if (debug && (error = CubDebug(cudaThreadSynchronize()))) break;

			// Set downsweep shared mem bank mode
			if (downsweep_props.sm_bank_config != spine_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(downsweep_props.sm_bank_config);

			// Downsweep scan from spine
			downsweep_props.kernel_func<<<grid_size[2], downsweep_props.cta_threads, dynamic_smem[2], stream>>>(
				d_bins[selector ^ 1],
				d_spine,
				d_keys[selector],
				d_keys[selector ^ 1],
				d_values[selector],
				d_values[selector ^ 1],
				current_bit,
				work);
			if (debug && (error = CubDebug(cudaThreadSynchronize()))) break;

			// Restore smem bank mode
			if (old_sm_config != downsweep_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(old_sm_config);

			// Update selector
			selector ^= 1;

		} while(0);		this->debug 			= debug;


		return error;
	}


	/**
	 * Dispatch hybrid partitioning pass
	 */
	cudaError_t DispatchHybridPass(
		int grid_size,
		int iteration)
	{
		cudaError_t error = cudaSuccess;
		do
		{
			// Print debug info
			if (debug)
			{
				printf("Hybrid pass: occupancy(%d), grid_size(%d), threads(%d)\n",
					hybrid_props.max_cta_occupancy, grid_size, hybrid_props.cta_threads);
				fflush(stdout);
			}

			// Set shared mem bank mode
			cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			if (old_sm_config != hybrid_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(hybrid_props.sm_bank_config);

			// Tile sorting kernel
			hybrid_props.kernel_func<<<grid_size, hybrid_props.cta_threads, 0, stream>>>(
				d_bins[selector],
				d_bins[selector ^ 1],
				d_keys[selector],
				d_keys[selector ^ 1],
				d_keys[0],
				d_values[selector],
				d_values[selector ^ 1],
				d_values[0],
				low_bit,
				iteration);

			if (debug && (error = CubDebug(cudaThreadSynchronize()))) break;

			// Restore smem bank mode
			if (old_sm_config != hybrid_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(old_sm_config);

			// Update selector
			selector ^= 1;

		} while(0);

		return error;
	}



	/**
	 * Dispatch single-CTA tile sort
	 */
	cudaError_t DispatchTile()
	{
		cudaError_t error = cudaSuccess;

		do {

			// Compute grid size
			int grid_size = 1;

			// Print debug info
			if (debug)
			{
				printf("Single tile: tile size(%d), occupancy(%d), grid_size(%d), threads(%d)\n",
					single_tile_props.tile_items,
					single_tile_props.max_cta_occupancy,
					grid_size,
					single_tile_props.cta_threads);
				fflush(stdout);
			}

			// Set shared mem bank mode
			cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			if (old_sm_config != single_tile_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(single_tile_props.sm_bank_config);

			// Single-CTA sorting kernel
			single_tile_props.kernel_func<<<grid_size, single_tile_props.cta_threads, 0, stream>>>(
				d_keys[selector],
				d_keys[selector],
				d_values[selector],
				d_values[selector],
				low_bit,
				num_bits,
				num_elements);

			if (debug && (error = CubDebug(cudaThreadSynchronize()))) break;

			// Restore smem bank mode
			if (old_sm_config != single_tile_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(old_sm_config);

		} while(0);

		return error;
	}


	/**
	 * Sort
	 */
	cudaError_t Sort(
		KeyType				*d_keys_in,
		ValueType			*d_values_in,
		SizeT				num_elements,
		int 				low_bit,
		int					num_bits,
		cudaStream_t		stream,
		int			 		max_grid_size)
	{
		this->d_keys[0] 		= d_keys_in;
		this->d_values[0] 		= d_values_in;
		this->num_elements 		= num_elements;
		this->low_bit 			= low_bit;
		this->num_bits 			= num_bits;
		this->stream 			= stream;
		this->max_grid_size 	= max_grid_size;

		cudaError_t error = cudaSuccess;
		do
		{
			if (num_elements <= single_tile_props.tile_items)
			{
				// Single tile sort
				error = DispatchTile();
				if (CubDebug(error)) break;
			}
			else
			{
				// Allocate temporary keys and values arrays
				error = allocator->Allocate((void**) &d_keys[1], sizeof(KeyType) * num_elements);
				if (CubDebug(error)) break;
				if (d_values_in != NULL)
				{
					error = allocator->Allocate((void**) &d_values[1], sizeof(ValueType) * num_elements);
					if (CubDebug(error)) break;
				}

				// Allocate and initialize partition descriptor queues
				int max_partitions = 32 * 32 * 32;
	//			int max_partitions = (num_elements + single_tile_props.tile_items - 1) / single_tile_props.tile_items;
				size_t partition_queue_bytes = sizeof(radix_sort::BinDescriptor) * max_partitions;

				error = allocator->Allocate((void**) &d_bins[0], partition_queue_bytes);
				if (CubDebug(error)) break;
				error = allocator->Allocate((void**) &d_bins[1], partition_queue_bytes);
				if (CubDebug(error)) break;
	/*
				error = cudaMemset(d_bins[0], 0, partition_queue_bytes);
				if (CubDebug(error)) break;
				error = cudaMemset(d_bins[1], 0, partition_queue_bytes);
				if (CubDebug(error)) break;
	*/
				// Dispatch first pass
				error = DispatchGlobalPass();
				if (CubDebug(error)) break;

				if (num_bits <= upsweep_props.radix_bits)
				{
					// Copy output into source buffer and be done
					error = cudaMemcpy(d_keys[0], d_keys[1], sizeof(KeyType) * num_elements, cudaMemcpyDeviceToDevice);
					if (CubDebug(error)) break;

					return error;
				}

				// Perform block iterations
				int grid_size = 32;
				{
					error = DispatchHybridPass(grid_size, 1);
					if (CubDebug(error)) break;

					grid_size *= 32;
				}
				if (num_elements > 128 * 16 * 32)
				{
					error = DispatchHybridPass(grid_size, 2);
					if (CubDebug(error)) break;

					grid_size *= 32;
				}
				if (num_elements > 128 * 16 * 32 * 32)
				{
					error = DispatchHybridPass(grid_size, 3);
					if (CubDebug(error)) break;

					grid_size *= 32;
				}
			}

		} while(0);

		return error;
	}

};


/**
 * Enact a large-problem sort.
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <
	typename KeyType,
	typename ValueType>
cudaError_t GpuRadixSortLarge(
	KeyType 		*d_keys,
	ValueType		*d_values,
	int 			num_elements,
	int				low_bit,
	int 			num_bits,
	cudaStream_t	stream 			= 0,
	int 			max_grid_size 	= 0,
	bool 			debug 			= false)
{
	if (num_elements <= 1)
	{
		// Nothing to do
		return cudaSuccess;
	}

	cudaError_t error = cudaSuccess;
	do
	{
		const radix_sort::ProblemSize PROBLEM_SIZE = radix_sort::LARGE_PROBLEM;

		// Initialize CUDA props
		cub::CudaProps cuda_props;
		error = cuda_props.Init();
		if (CubDebug(error)) break;

		// Get default allocator
		cub::CachedAllocator *allocator = cub::CubCachedAllocator<void>();

		// Construct and configure problem instance
		GpuRadixSort<cub::CachedAllocator, KeyType, ValueType, int> problem_instance(
			&cuda_props,
			allocator,
			debug);

//		if (cuda_props.ptx_version >= 200)
		{
			error = problem_instance.template Configure<200, PROBLEM_SIZE>();
			if (CubDebug(error)) break;
		}
/*
		else if (cuda_props.ptx_version >= 130)
		{
			error = problem_instance.template Configure<130, PROBLEM_SIZE>();
			if (CubDebug(error)) break;
		}
		else
		{
			error = problem_instance.template Configure<100, PROBLEM_SIZE>();
			if (CubDebug(error)) break;
		}
*/
		// Sort
		error = problem_instance.Sort(
			d_keys,
			d_values,
			num_elements,
			low_bit,
			num_bits,
			stream,
			max_grid_size);
		if (CubDebug(error)) break;

	} while (0);

	return error;
}





} // namespace back40
BACK40_NS_POSTFIX
