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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Management of temporary device storage needed for maintaining partial
 * reductions between subsequent grids
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace util {

/**
 * Manages device storage needed for communicating partial reductions
 * between CTAs in subsequent grids
 */
class Spine
{
protected :

	// Temporary spine storage
	void *d_spine;

	// Number of bytes backed by d_spine
	size_t spine_bytes;

	// GPU d_spine was allocated on
	int d_spine_gpu;


public :

	/**
	 * Constructor
	 */
	Spine() :
		d_spine(NULL),
		spine_bytes(0),
		d_spine_gpu(B40C_INVALID_DEVICE) {}


	/**
	 * Deallocates and resets the spine
	 */
	void HostReset()
	{
		if (d_spine_gpu != B40C_INVALID_DEVICE) {

			int current_gpu;
			cudaGetDevice(&current_gpu);

			// Deallocate
			cudaSetDevice(d_spine_gpu);
			util::B40CPerror(cudaFree(d_spine), "Spine cudaFree d_spine failed: ", __FILE__, __LINE__);
			d_spine = NULL;
			d_spine_gpu = -1;

			cudaSetDevice(current_gpu);
		}
		spine_bytes = 0;
	}


	/**
	 * Destructor
	 */
	virtual ~Spine()
	{
		HostReset();
	}


	/**
	 * Getter
	 */
	void* operator()()
	{
		return d_spine;
	}


	/**
	 * Sets up the spine to accommodate partials of the specified type
	 * produced/consumed by grids of the specified sweep grid size (lazily
	 * allocating it if necessary)
	 *
	 * Grows as necessary.
	 */
	template <typename T>
	cudaError_t Setup(int sweep_grid_size, int spine_elements)
	{
		cudaError_t retval = cudaSuccess;
		do {
			size_t problem_spine_bytes = spine_elements * sizeof(T);

			if (problem_spine_bytes > spine_bytes) {

				// Deallocate if exists
				HostReset();

				// Reallocate
				cudaGetDevice(&d_spine_gpu);
				spine_bytes = problem_spine_bytes;
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
					"Spine cudaMalloc d_spine failed", __FILE__, __LINE__)) break;
			}
		} while (0);

		return retval;
	}
};

} // namespace util
} // namespace b40c

