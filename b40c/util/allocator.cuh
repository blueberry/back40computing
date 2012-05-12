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
 * Work Management Datastructures
 ******************************************************************************/

#pragma once

#include <math.h>
#include <set>
#include <map>

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/ns_umbrella.cuh>
#include <b40c/util/spinlock.cuh>

B40C_NS_PREFIX
namespace b40c {
namespace util {

struct CachedAllocator
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	enum
	{
		LOG8_MIN 				= 3, 		// Minimum bin size: 512 bytes
		INVALID_GPU_ORDINAL		= -1,
	};


	// Block descriptor
	struct BlockDescriptor
	{
		int			gpu;		// GPU ordinal
		void* 		d_ptr;		// Device pointer
		int			bin;		// Bin category
		size_t		bytes;		// Size of allocation in bytes

		static int ComputeBin(size_t bytes)
		{
			return (bytes > 0) ?
				CUB_MIN(LOG8_MIN, log(bytes) / log(8)) :
				LOG8_MIN;
		}

		// Constructor
		BlockDescriptor(int gpu, size_t bytes, void *d_ptr) :
			gpu(gpu),
			d_ptr(d_ptr),
			bin(ComputeBin(bytes)),
			bytes(bytes) {}

		// Constructor
		BlockDescriptor(int gpu, void *d_ptr) :
			gpu(gpu),
			d_ptr(d_ptr),
			bin(0),
			bytes(0) {}

		// Constructor
		BlockDescriptor(int gpu, size_t bytes) :
			gpu(gpu),
			d_ptr(NULL),
			bin(ComputeBin(bytes)),
			bytes(bytes) {}

		// Comparison functor for comparing device pointers
		static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
		{
			if (a.gpu < b.gpu) {
				return true;
			} else if (a.gpu > b.gpu) {
				return false;
			} else {
				return (a.d_ptr < b.d_ptr);
			}
		}

		// Comparison functor for comparing allocation sizes
		static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
		{
			if (a.gpu < b.gpu) {
				return true;
			} else if (a.gpu > b.gpu) {
				return false;
			} else {
				return (a.bytes < b.bytes);
			}
		}
	};

	// Set type for free blocks (ordered by size)
	typedef std::set<BlockDescriptor, BlockDescriptor::SizeCompare> FreeBlocks;

	// Set type for used blocks (ordered by ptr)
	typedef std::set<BlockDescriptor, BlockDescriptor::PtrCompare> UsedBlocks;

	// Map type of gpu ordinals to the number of free bytes cached by each
	typedef std::map<int, size_t> FreeBytes;


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	SpinLock 		spin_lock;
	FreeBlocks 		free_blocks;
	UsedBlocks 		used_blocks;

	size_t 			max_free_bytes;
	FreeBytes		free_bytes;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor.  Sets the limit on the number of unused device bytes
	 * this allocator is allowed to cache on each GPU
	 */
	CachedAllocator(size_t max_free_bytes = (1024 * 1024)) :
		spin_lock(0),
		max_free_bytes(max_free_bytes)
	{}


	/**
	 * Sets the limit on the number of unused device bytes this allocator
	 * is allowed to cache.
	 */
	void SetMaxFreeBytes(size_t max_free_bytes)
	{
		// Lock
		Lock(&spin_lock);

		this->max_free_bytes = max_free_bytes;

		// Unlock
		Unlock(&spin_lock);
	}


	/**
	 * Return a suitable allocation of device memory for the given size
	 * on the specified GPU ordinal
	 */
	cudaError_t Allocate(int gpu, void** d_ptr, size_t bytes)
	{
		bool locked 			= false;
		int previous_gpu 		= INVALID_GPU_ORDINAL;
		cudaError_t error 		= cudaSuccess;

		BlockDescriptor search_key(gpu, bytes);

		// Lock
		if (!locked) {
			Lock(&spin_lock);
			locked = true;
		}

		do {
			// Find a free block big enough within the same bin
			FreeBlocks::iterator block_itr = free_blocks.upper_bound(search_key);

			if ((block_itr != free_blocks.end()) &&
				(block_itr->gpu == gpu) &&
				(block_itr->bin == search_key.bin))
			{
				// Reuse existing cache block.  Insert into used blocks.
				search_key = *block_itr;
				used_blocks.insert(search_key);

				// Remove from free blocks
				free_blocks.erase(block_itr);
				free_bytes[gpu] -= search_key.bytes;
			}
			else
			{
				// Need to allocate a new cache block. Unlock.
				if (locked) {
					Unlock(&spin_lock);
					locked = false;
				}

				// Set to specified GPU
				error = cudaGetDevice(&previous_gpu);
				if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__)) break;
				error = cudaSetDevice(gpu);
				if (util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__)) break;

				// Allocate
				error = cudaMalloc(&search_key.d_ptr, bytes);
				if (util::B40CPerror(error, "cudaMalloc failed ", __FILE__, __LINE__)) break;

				// Lock
				if (!locked) {
					Lock(&spin_lock);
					locked = true;
				}

				// Insert into used blocks
				used_blocks.insert(search_key);
			}
		} while(0);

		// Unlock
		if (locked) {
			Unlock(&spin_lock);
			locked = false;
		}

		// Attempt to revert back to previous GPU if necessary
		if (previous_gpu != INVALID_GPU_ORDINAL)
		{
			error = cudaSetDevice(previous_gpu);
			util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__);
		}

		// Copy device pointer to output parameter
		*d_ptr = search_key.d_ptr;

		return error;
	}


	/**
	 * Return a suitable allocation of device memory for the given size
	 * on the current GPU ordinal
	 */
	cudaError_t Allocate(void** d_ptr, size_t bytes)
	{
		int current_gpu;
		cudaError_t error = cudaGetDevice(&current_gpu);
		if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__))
			return error;

		return Allocate(current_gpu, d_ptr, bytes, current_gpu);
	}


	/**
	 * Return a used allocation of GPU memory on the specified gpu
	 * ordinal to the allocator
	 */
	cudaError_t Deallocate(int gpu, void* d_ptr)
	{
		bool locked 			= false;
		cudaError_t error 		= cudaSuccess;

		BlockDescriptor search_key(gpu, d_ptr);

		// Lock
		if (!locked) {
			Lock(&spin_lock);
			locked = true;
		}

		do {
			// Find corresponding block descriptor
			UsedBlocks::iterator block_itr = used_blocks.find(search_key);
			if (block_itr == used_blocks.end())
			{
				// Cannot find pointer
				error = util::B40CPerror(cudaErrorUnknown, "Deallocate failed ", __FILE__, __LINE__);
				break;
			}
			else
			{
				// Remove from used blocks
				search_key = *block_itr;
				used_blocks.erase(block_itr);

				// Insert into free blocks if we can fit it
				if (free_blocks[gpu] + search_key.bytes <= max_free_bytes)
				{
					free_blocks.insert(search_key);
					free_bytes[gpu] += search_key.bytes;
				}
			}
		} while (0);

		// Unlock
		if (locked) {
			Unlock(&spin_lock);
			locked = false;
		}

		return error;
	}


	/**
	 * Return a used allocation of device memory on the current GPU
	 * ordinal to the allocator
	 */
	cudaError_t Deallocate(void* d_ptr)
	{
		int current_gpu;
		cudaError_t error = cudaGetDevice(&current_gpu);
		if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__))
			return error;

		return Deallocate(current_gpu, d_ptr);
	}


	cudaError_t FreeAllCached()
	{
		cudaError_t error 		= cudaSuccess;
		bool locked 			= false;
		int previous_gpu 		= INVALID_GPU_ORDINAL;
		int current_gpu			= INVALID_GPU_ORDINAL;

		// Lock
		if (!locked) {
			Lock(&spin_lock);
			locked = true;
		}

		while (!free_blocks.empty())
		{
			// Get first block
			FreeBlocks::iterator begin = free_blocks.begin();

			// Get original GPU ordinal if necessary
			if (previous_gpu == INVALID_GPU_ORDINAL)
			{
				error = cudaGetDevice(&previous_gpu);
				if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__)) break;
			}

			// Set current GPU ordinal if necessary
			if (begin->gpu != current_gpu)
			{
				error = cudaSetDevice(begin->gpu);
				if (util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__)) break;
				current_gpu = begin->gpu;
			}

			// Free device memory
			error = cudaFree(begin->d_ptr);
			if (util::B40CPerror(error, "cudaGetDevice failed ", __FILE__, __LINE__)) break;

			// Reduce balance and erase entry
			free_bytes[current_gpu] -= begin->bytes;
			free_blocks.erase(begin);
		}

		// Unlock
		if (locked) {
			Unlock(&spin_lock);
			locked = false;
		}

		// Attempt to revert back to previous GPU if necessary
		if (previous_gpu != INVALID_GPU_ORDINAL)
		{
			error = cudaSetDevice(previous_gpu);
			util::B40CPerror(error, "cudaSetDevice failed ", __FILE__, __LINE__);
		}

		return error;
	}
};




} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
