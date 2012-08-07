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
 * Management of temporary device storage
 ******************************************************************************/

#pragma once

#include "../util/error_utils.cuh"
#include "../util/cuda_properties.cuh"
#include "../util/ns_umbrella.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace util {

/**
 * Manages device storage
 */
struct Scratch
{
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Device scratch storage
	void *d_scratch;

	// Host-mapped scratch storage (if so constructed)
	void *h_scratch;

	// Number of bytes backed by d_scratch
	size_t scratch_bytes;

	// GPU d_scratch was allocated on
	int gpu;

	// Whether or not the scratch has a shadow scratch on the host
	bool host_shadow;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor (device-allocated scratch)
	 */
	Scratch() :
		d_scratch(NULL),
		h_scratch(NULL),
		scratch_bytes(0),
		gpu(CUB_INVALID_DEVICE),
		host_shadow(false) {}


	/**
	 * Constructor
	 *
	 * @param host_shadow
	 * 		Whether or not the scratch has a shadow scratch on the host
	 */
	Scratch(bool host_shadow) :
		d_scratch(NULL),
		h_scratch(NULL),
		scratch_bytes(0),
		gpu(CUB_INVALID_DEVICE),
		host_shadow(host_shadow) {}


	/**
	 * Deallocates and resets the scratch
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		do {

			if (gpu == CUB_INVALID_DEVICE) return retval;

			// Save current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Scratch cudaGetDevice failed: ", __FILE__, __LINE__)) break;
#if CUDA_VERSION >= 4000
			if (retval = util::B40CPerror(cudaSetDevice(gpu),
				"Scratch cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif
			if (d_scratch) {
				// Deallocate
				if (retval = util::B40CPerror(cudaFree(d_scratch),
					"Scratch cudaFree d_scratch failed: ", __FILE__, __LINE__)) break;
				d_scratch = NULL;
			}
			if (h_scratch) {
				// Deallocate
				if (retval = util::B40CPerror(cudaFreeHost((void *) h_scratch),
					"Scratch cudaFreeHost h_scratch failed", __FILE__, __LINE__)) break;

				h_scratch = NULL;
			}

#if CUDA_VERSION >= 4000
			// Restore current gpu
			if (retval = util::B40CPerror(cudaSetDevice(current_gpu),
				"Scratch cudaSetDevice failed: ", __FILE__, __LINE__)) break;
#endif

			gpu 			= CUB_INVALID_DEVICE;
			scratch_bytes	 	= 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~Scratch()
	{
		HostReset();
	}


	/**
	 * Device scratch storage accessor
	 */
	void* operator()()
	{
		return d_scratch;
	}


	/**
	 * Sets up the scratch to accommodate the specified number of bytes.
	 * Reallocates if necessary.
	 */
	template <typename SizeT>
	cudaError_t Setup(SizeT problem_scratch_bytes)
	{
		cudaError_t retval = cudaSuccess;
		do {
			// Get current gpu
			int current_gpu;
			if (retval = util::B40CPerror(cudaGetDevice(&current_gpu),
				"Scratch cudaGetDevice failed: ", __FILE__, __LINE__)) break;

			// Check if big enough and if lives on proper GPU
			if ((problem_scratch_bytes > scratch_bytes) || (gpu != current_gpu)) {

				// Deallocate if exists
				if (retval = HostReset()) break;

				// Remember device
				gpu = current_gpu;

				// Reallocate
				scratch_bytes = problem_scratch_bytes;

				// Allocate on device
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_scratch, scratch_bytes),
					"Scratch cudaMalloc d_scratch failed", __FILE__, __LINE__)) break;

				if (host_shadow)
				{
					// Allocate pinned memory for h_scratch
					int flags = cudaHostAllocMapped;
					if (retval = util::B40CPerror(cudaHostAlloc((void **)&h_scratch, problem_scratch_bytes, flags),
						"Scratch cudaHostAlloc h_scratch failed", __FILE__, __LINE__)) break;
				}
			}
		} while (0);

		return retval;
	}


	/**
	 * Syncs the shadow host scratch with device scratch
	 */
	cudaError_t Sync(cudaStream_t stream)
	{
		return cudaMemcpyAsync(
			h_scratch,
			d_scratch,
			scratch_bytes,
			cudaMemcpyDeviceToHost,
			stream);
	}


	/**
	 * Syncs the shadow host scratch with device scratch
	 */
	cudaError_t Sync()
	{
		return cudaMemcpy(
			h_scratch,
			d_scratch,
			scratch_bytes,
			cudaMemcpyDeviceToHost);
	}


};

} // namespace util
} // namespace b40c
B40C_NS_POSTFIX
