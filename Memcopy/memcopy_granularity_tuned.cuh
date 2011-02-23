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
 * Tuned Memcopy Granularity Types
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"

#include "memcopy_api_granularity.cuh"
#include "memcopy_kernel.cuh"

namespace b40c {
namespace memcopy {


/******************************************************************************
 * Tuning classifiers to specialize granularity types on
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProblemSize
{
	SMALL 	= 0,
	LARGE 	= 1
};


/**
 * Enumeration of architecture-families that we have tuned for below
 */
enum Family
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Classifies a given CUDA_ARCH into an architecture-family
 */
template <int CUDA_ARCH>
struct FamilyClassifier
{
	static const Family FAMILY =	(CUDA_ARCH < SM13) ? 	SM10 :
									(CUDA_ARCH < SM20) ? 	SM13 :
															SM20;
};


/******************************************************************************
 * Granularity tuning types
 *
 * Specialized by family (and optionally by specific architecture) and by
 * problem size type
 ******************************************************************************/

/**
 * Default, catch-all granularity parameterization type.  Defers to the
 * architecture "family" that we know we have specialization type(s) for below.
 */
template <int CUDA_ARCH, ProblemSize PROBLEM_SIZE>
struct TunedConfig : TunedConfig<FamilyClassifier<CUDA_ARCH>::FAMILY, PROBLEM_SIZE> {};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

template <>
struct TunedConfig<SM20, LARGE>
	: MemcopyKernelConfig<unsigned int, size_t, 8, 7, 1, 1, CG, true, 9>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

template <>
struct TunedConfig<SM20, SMALL>
	: MemcopyKernelConfig<unsigned long long, size_t, 8, 5, 1, 1, CG, false, 7>
{
	static const ProblemSize PROBLEM_SIZE = SMALL;
};


//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

template <>
struct TunedConfig<SM13, LARGE>
	: MemcopyKernelConfig<unsigned short, size_t, 8, 7, 2, 0, NONE, false, 9>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

template <>
struct TunedConfig<SM13, SMALL>
	: MemcopyKernelConfig<unsigned short, size_t, 8, 5, 2, 0, NONE, false, 7>
{
	static const ProblemSize PROBLEM_SIZE = SMALL;
};


//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------

template <ProblemSize _PROBLEM_SIZE>
struct TunedConfig<SM10, _PROBLEM_SIZE>
: MemcopyKernelConfig<unsigned short, size_t, 8, 5, 2, 0, NONE, false, 7>
{
	static const ProblemSize PROBLEM_SIZE = _PROBLEM_SIZE;
};






/******************************************************************************
 * Memcopy kernel entry points that can derive a tuned granularity type
 * implicitly from the PROBLEM_SIZE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

template <int PROBLEM_SIZE, typename SizeT>
__launch_bounds__ (
	(1 << TunedConfig<__B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::MemcopyKernelConfig::LOG_THREADS),
	(TunedConfig<__B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::MemcopyKernelConfig::CTA_OCCUPANCY))
__global__ void TunedMemcopyKernel(
	void			* __restrict d_out,
	void			* __restrict d_in,
	SizeT 			* __restrict d_work_progress,
	CtaWorkDistribution<SizeT> work_decomposition,
	int 			progress_selector,
	int 			extra_bytes)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef TunedConfig<__B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE> Config;
	typedef typename Config::T T;

	T* out = (T*)(d_out);
	T* in = (T*)(d_in);

	// Invoke the wrapped kernel logic
	MemcopyPass<Config, Config::WORK_STEALING>::Invoke(
		out, in, d_work_progress, work_decomposition, progress_selector, extra_bytes);
}



}// namespace memcopy
}// namespace b40c

