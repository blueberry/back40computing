/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
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
 * Debug error display routines
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {


// CUB debugging macro (prints error messages to stderr)
#ifndef CUB_STDERR
#if (defined(__THRUST_SYNCHRONOUS) || defined(DEBUG) || defined(_DEBUG))
	#define CUB_STDERR	(1)
#else
	#define CUB_STDERR	(0)
#endif
#endif


/**
 * If print is true and the specified CUDA error is not cudaSuccess, the corresponding
 * error message is printed to stderr along with the supplied source context.  Returns
 * the CUDA error.
 */
__host__ __device__ __forceinline__ cudaError_t Debug(
	cudaError_t error,
	const char *message,
	const char *filename,
	int line)
{
	if (CUB_STDERR) {
		fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}


/**
 * If print is true and the specified CUDA error is not cudaSuccess, the corresponding
 * error message is printed to stderr.  Returns the CUDA error.
 */
__host__ __device__ __forceinline__ cudaError_t Debug(cudaError_t error)
{
	if (CUB_STDERR) {
		fprintf(stderr, "(CUDA error %d: %s)\n", error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}



} // namespace cub
CUB_NS_POSTFIX
