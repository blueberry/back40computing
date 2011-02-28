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
 * Base class for dynamic architecture dispatch
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>

namespace b40c {


/**
 * Specialization for the device compilation-path. Use in accordance
 * with the curiously-recurring template pattern (CRTP).
 *
 * Dispatches to an Enact() method in the derived class that is
 * specialized by CUDA_ARCH.  This path drives the actual compilation of
 * kernels, allowing them to be specific to CUDA_ARCH.
 */
template <int CUDA_ARCH, typename Derived>
class ArchDispatch
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
 * Specialization specialization for the host compilation-path. Use in accordance
 * with the curiously-recurring template pattern (CRTP).
 *
 * Dispatches to an Enact() method in the derived class that is specialized by
 * the version of the accompanying PTX assembly.  This path does not drive the
 * compilation of kernels.
 */
template <typename Derived>
class ArchDispatch<0, Derived>
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

