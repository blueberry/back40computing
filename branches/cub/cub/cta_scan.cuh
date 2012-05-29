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

#include <cub/device_props.cuh>
#include <cub/type_utils.cuh>
#include <cub/operators.cuh>
#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {


/**
 * Cooperative prefix scan abstraction for CTAs.
 *
 * Features:
 * 		- Very efficient (only two synchronization barriers).
 * 		- Zero bank conflicts for most types.
 * 		- Supports non-commutative scan operators.
 * 		- Supports scan over strip-mined CTA tiles.  (For a given tile of
 * 			input, each thread acquires STRIPS "strips" of ELEMENTS consecutive
 * 			inputs, where the stride between a given thread's strips is
 * 			(ELEMENTS * CTA_THREADS) elements.)
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a multiple of the warp size
 * 		- The scan type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	int 		CTA_THREADS,			// The CTA size in threads
	typename 	T,						// The scan type
	int 		CTA_STRIPS = 1>			// When strip-mining, the number of CTA-strips per tile
struct CtaScan
{

};



} // namespace cub
CUB_NS_POSTFIX
