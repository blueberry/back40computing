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
 * Cooperative scan abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Policy for default radix bits (based upon key type and the target
 * architecture).
 */
template <typename KeyType, int TARGET_ARCH>
struct RadixBitsPolicy
{
	enum
	{
		VALUE = 5,
	};
};



/**
 * Cooperative radix sorting abstraction for CTAs.
 */
template <
	int 					CTA_THREADS,
	int 					BITS_REMAINING,
	int 					CURRENT_BIT,
	typename 				KeyType,
	typename 				ValueType 			= NullType,
	int 					RADIX_BITS			= DefaultRadixBits<KeyType, PTX_ARCH>::VALUE,
	cudaSharedMemConfig 	SMEM_CONFIG 		= cudaSharedMemBankSizeFourByte>				// Shared memory bank size
struct CtaRadixSort
{


};



} // namespace cub
CUB_NS_POSTFIX
