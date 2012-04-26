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
 * Kernel utilities for reading/writing memory using cache modifiers
 ******************************************************************************/

#pragma once

namespace cub {

/**
 * Enumeration of read cache modifiers.
 */
enum ReadModifier {
	READ_NONE,		// Default (currently READ_CA)
	READ_CA,		// Cache at all levels
	READ_CG,		// Cache at global level
	READ_CS, 		// Cache streaming (likely to be accessed once)
	READ_CV, 		// Cache as volatile (including cached system lines)
	READ_TEX,		// Texture (defaults to NONE if no tex reference is provided)

	READ_LIMIT
};


/**
 * Enumeration of write cache modifiers.
 */
enum WriteModifier {
	WRITE_NONE,		// Default (currently WRITE_WB)
	WRITE_WB,		// Cache write-back all coherent levels
	WRITE_CG,		// Cache at global level
	WRITE_CS, 		// Cache streaming (likely to be accessed once)
	WRITE_WT, 		// Cache write-through (to system memory)

	WRITE_LIMIT
};


} // namespace cub

