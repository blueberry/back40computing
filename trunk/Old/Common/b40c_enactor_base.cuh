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
 * Base Enactor Classes
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {


/******************************************************************************
 * Enactor base class
 ******************************************************************************/

class EnactorBase
{
public:

	//---------------------------------------------------------------------
	// Utility Fields
	//---------------------------------------------------------------------

	// Debug level.  If set, the enactor blocks after kernel calls to check
	// for successful launch/execution
	bool DEBUG;


	// The arch version of the code for the current device that actually have
	// compiled kernels for
	int PtxVersion()
	{
		return this->cuda_props.kernel_ptx_version;
	}

protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Device properties
	const util::CudaProperties cuda_props;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Returns the number of threadblocks to launch for the given problem size.
	 * Does not exceed the full-occupancy on the current device or the
	 * optional max_grid_size limit.
	 *
	 * Useful for kernels that work-steal or use global barriers (where
	 * over-subscription is not ideal or allowed)
	 */
	template <int SCHEDULE_GRANULARITY, int CTA_OCCUPANCY>
	int OccupiedGridSize(int num_elements, int max_grid_size = 0)
	{
		int grid_size = cuda_props.device_props.multiProcessorCount * CTA_OCCUPANCY;
		if ((max_grid_size > 0) && (grid_size > max_grid_size)) {
			grid_size = max_grid_size;
		}

		// Reduce if we have less work than we can divide up among this
		// many CTAs
		int grains = (num_elements + SCHEDULE_GRANULARITY - 1) / SCHEDULE_GRANULARITY;
		if (grid_size > grains) {
			grid_size = grains;
		}

		// Reduce by override, if specified
		if (max_grid_size > 0) {
			grid_size = B40C_MIN(max_grid_size, grid_size);
		}

		return grid_size;
	}


	/**
	 * Returns the number of threadblocks to launch for the given problem size.
	 * May over/under subscribe the current device based upon heuristics.  Does not
	 * the optional max_grid_size limit.
	 *
	 * Useful for kernels that evenly divide up the work amongst threadblocks.
	 */
	template <int SCHEDULE_GRANULARITY, int CTA_OCCUPANCY>
	int OversubscribedGridSize(int num_elements, int max_grid_size)
	{
		int grid_size;
		if (cuda_props.device_sm_version < 120) {

			// G80/G90: CTA occupancy times SM count
			grid_size = cuda_props.device_props.multiProcessorCount * CTA_OCCUPANCY;

		} else if (cuda_props.device_sm_version < 200) {

			// GT200: Special sauce

			// Start with with full downsweep occupancy of all SMs
			grid_size =
				cuda_props.device_props.multiProcessorCount * CTA_OCCUPANCY;

			// Increase by default every 64 million key-values
			int step = 1024 * 1024 * 64;
			grid_size *= (num_elements + step - 1) / step;

			double multiplier1 = 4.0;
			double multiplier2 = 16.0;

			double delta1 = 0.068;
			double delta2 = 0.1285;

			int dividend = (num_elements + 512 - 1) / 512;

			int bumps = 0;
			while(true) {

				if (grid_size <= cuda_props.device_props.multiProcessorCount) {
					break;
				}

				double quotient = ((double) dividend) / (multiplier1 * grid_size);
				quotient -= (int) quotient;

				if ((quotient > delta1) && (quotient < 1 - delta1)) {

					quotient = ((double) dividend) / (multiplier2 * grid_size / 3.0);
					quotient -= (int) quotient;

					if ((quotient > delta2) && (quotient < 1 - delta2)) {
						break;
					}
				}

				if (bumps == 3) {
					// Bump it down by 27
					grid_size -= 27;
					bumps = 0;
				} else {
					// Bump it down by 1
					grid_size--;
					bumps++;
				}
			}

		} else {

			// GF10x
			if (cuda_props.device_sm_version == 210) {
				// GF110
				grid_size = 4 * (cuda_props.device_props.multiProcessorCount * CTA_OCCUPANCY);
			} else {
				// Anything but GF110
				grid_size = 4 * (cuda_props.device_props.multiProcessorCount * CTA_OCCUPANCY) - 2;
			}
		}

		// Reduce if we have less work than we can divide up among this
		// many CTAs
		int grains = (num_elements + SCHEDULE_GRANULARITY - 1) / SCHEDULE_GRANULARITY;
		if (grid_size > grains) {
			grid_size = grains;
		}

		// Reduce by override, if specified
		if (max_grid_size > 0) {
			grid_size = B40C_MIN(max_grid_size, grid_size);
		}

		return grid_size;
	}


	//---------------------------------------------------------------------
	// Constructors
	//---------------------------------------------------------------------

	EnactorBase() :
#if	defined(__THRUST_SYNCHRONOUS) || defined(DEBUG) || defined(_DEBUG)
			DEBUG(true)
#else
			DEBUG(false)
#endif
		{}

};




/******************************************************************************
 * Architecture base class
 ******************************************************************************/

/**
 * Enactor specialization for the device compilation-path. Use in accordance
 * with the curiously-recurring template pattern (CRTP).
 *
 * Dispatches to an Enact() method in the derived class that is
 * specialized by CUDA_ARCH.  This path drives the actual compilation of
 * kernels, allowing them to be specific to CUDA_ARCH.
 */
template <int CUDA_ARCH, typename Derived>
class Architecture
{
protected:

	template<typename Storage, typename Detail>
	cudaError_t Enact(Storage &problem_storage, Detail &detail)
	{
		Derived *enactor = static_cast<Derived*>(this);
		return enactor->template Enact<CUDA_ARCH, Storage, Detail>(problem_storage, detail);
	}
};


/**
 * Enactor specialization for the host compilation-path. Use in accordance
 * with the curiously-recurring template pattern (CRTP).
 *
 * Dispatches to an Enact() method in the derived class that is specialized by
 * the version of the accompanying PTX assembly.  This path does not drive the
 * compilation of kernels.
 */
template <typename Derived>
class Architecture<0, Derived>
{
protected:

	template<typename Storage, typename Detail>
	cudaError_t Enact(Storage &problem_storage, Detail &detail)
	{
		// Determine the arch version of the we actually have a compiled kernel for
		Derived *enactor = static_cast<Derived*>(this);

		// Dispatch
		switch (enactor->PtxVersion()) {
		case 100:
			return enactor->template Enact<100, Storage, Detail>(problem_storage, detail);
		case 110:
			return enactor->template Enact<110, Storage, Detail>(problem_storage, detail);
		case 120:
			return enactor->template Enact<120, Storage, Detail>(problem_storage, detail);
		case 130:
			return enactor->template Enact<130, Storage, Detail>(problem_storage, detail);
		case 200:
			return enactor->template Enact<200, Storage, Detail>(problem_storage, detail);
		case 210:
			return enactor->template Enact<210, Storage, Detail>(problem_storage, detail);
		default:
			// We were compiled for something new: treat it as we would SM2.0
			return enactor->template Enact<200, Storage, Detail>(problem_storage, detail);
		};
	}
};



} // namespace b40c

