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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Distribution sort downsweep tuning configuration.
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace downsweep {


/**
 * Downsweep tuning configuration.  This C++ type encapsulates our
 * kernel-tuning parameters (they are reflected via the static fields).
 *  
 * The kernels are specialized for problem-type, SM-version, etc. by declaring 
 * them with different performance-tuned parameterizations of this type.  By 
 * incorporating this type into the kernel code itself, we guide the compiler in 
 * expanding/unrolling the kernel code for specific architectures and problem 
 * types.
 * 
 * Constraints:
 * 		(i) 	A load can't contain more than 256 keys or we might overflow inside a lane of  
 * 				8-bit composite counters, i.e., (threads * load-vec-size <= 256), equivalently:
 * 
 * 					(LOG_THREADS + LOG_LOAD_VEC_SIZE <= 8)
 * 
 * 		(ii) 	We must have between one and one warp of raking threads per lane of composite 
 * 				counters, i.e., (1 <= raking-threads / (loads-per-cycle * radix-digits / 4) <= 32), 
 * 				equivalently:
 * 
 * 					(0 <= LOG_RAKING_THREADS - LOG_LOADS_PER_CYCLE - RADIX_BITS + 2 <= B40C_LOG_WARP_THREADS(arch))
 *     
 * 		(iii) 	We must have more than radix-digits threads in the threadblock,  
 * 				i.e., (threads >= radix-digits) equivalently:
 * 
 * 					LOG_THREADS >= RADIX_BITS
 */
template <
	// Problem type
	typename _KeyType,
	typename _ValueType,
	typename _SizeT,

	int _CUDA_ARCH,
	int _RADIX_BITS,
	int _LOG_SCHEDULE_GRANULARITY,
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_CYCLE,
	int _LOG_CYCLES_PER_TILE,
	int _LOG_RAKING_THREADS,
	util::io::ld::CacheModifier _READ_MODIFIER,
	util::io::st::CacheModifier _WRITE_MODIFIER,
	bool _EARLY_EXIT>

struct TuningConfig
{
	typedef _KeyType							KeyType;
	typedef _ValueType							ValueType;
	typedef _SizeT								SizeT;

	static const int CUDA_ARCH					= _CUDA_ARCH;
	static const int RADIX_BITS					= _RADIX_BITS;
	static const int LOG_SCHEDULE_GRANULARITY	= _LOG_SCHEDULE_GRANULARITY;
	static const int CTA_OCCUPANCY  			= _CTA_OCCUPANCY;
	static const int LOG_THREADS 				= _LOG_THREADS;
	static const int LOG_LOAD_VEC_SIZE 			= _LOG_LOAD_VEC_SIZE;
	static const int LOG_LOADS_PER_CYCLE		= _LOG_LOADS_PER_CYCLE;
	static const int LOG_CYCLES_PER_TILE		= _LOG_CYCLES_PER_TILE;
	static const int LOG_RAKING_THREADS			= _LOG_RAKING_THREADS;
	static const util::io::ld::CacheModifier READ_MODIFIER = _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER = _WRITE_MODIFIER;
	static const bool EARLY_EXIT				= _EARLY_EXIT;
};



} // namespace downsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

