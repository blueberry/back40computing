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
 * Inlined PTX intrinsics
 ******************************************************************************/

#pragma once

#include "ns_umbrella.cuh"

CUB_NS_PREFIX
namespace cub {


/******************************************************************************
 * Inlined PTX helper macros
 ******************************************************************************/

/**
 * Register modifier for pointer-types (for inlining PTX assembly)
 */
#if defined(_WIN64) || defined(__LP64__)
	#define __CUB_LP64__ 1
	// 64-bit register modifier for inlined asm
	#define _CUB_ASM_PTR_ "l"
#else
	#define __CUB_LP64__ 0
	// 32-bit register modifier for inlined asm
	#define _CUB_ASM_PTR_ "r"
#endif


/******************************************************************************
 * Inlined PTX intrinsics
 ******************************************************************************/

/**
 * Shift-right then add.  Returns (x >> shift) + addend.
 */
__device__ __forceinline__ unsigned int SHR_ADD(
	unsigned int x,
	unsigned int shift,
	unsigned int addend)
{
	unsigned int ret;
#if __CUDA_ARCH__ >= 200
	asm("vshr.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
		"=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
	ret = (x >> shift) + addend;
#endif
	return ret;
}


/**
 * Shift-left then add.  Returns (x << shift) + addend.
 */
__device__ __forceinline__ unsigned int SHL_ADD(
	unsigned int x,
	unsigned int shift,
	unsigned int addend)
{
	unsigned int ret;
#if __CUDA_ARCH__ >= 200
	asm("vshl.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
		"=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
	ret = (x << shift) + addend;
#endif
	return ret;
}


/**
 * Bitfield-extract.
 */
__device__ __forceinline__ unsigned int BFE(
	unsigned int source,
	unsigned int bit_start,
	unsigned int num_bits)
{
	unsigned int bits;
#if __CUDA_ARCH__ >= 200
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(bit_start), "r"(num_bits));
#else
	const unsigned int MASK = (1 << num_bits) - 1;
	bits = (source >> bit_start) & MASK;
#endif
	return bits;
}


/**
 * Bitfield insert.  Inserts the first num_bits of y into x starting at bit_start
 */
__device__ __forceinline__ void BFI(
	unsigned int &ret,
	unsigned int x,
	unsigned int y,
	unsigned int bit_start,
	unsigned int num_bits)
{
#if __CUDA_ARCH__ >= 200
	asm("bfi.b32 %0, %1, %2, %3, %4;" :
		"=r"(ret) : "r"(y), "r"(x), "r"(bit_start), "r"(num_bits));
#else
	// TODO
#endif
}


/**
 * Three-operand add
 */
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
#if __CUDA_ARCH__ >= 200
	asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(x) : "r"(x), "r"(y), "r"(z));
#else
	x = x + y + z;
#endif
	return x;
}


/**
 * Byte-permute. Pick four arbitrary bytes from two 32-bit registers, and
 * reassemble them into a 32-bit destination register
 */
__device__ __forceinline__ int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
	int ret;
	asm("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(index));
	return ret;
}


/**
 * Sync-threads barrier.
 */
__device__ __forceinline__ void BAR(int count)
{
	asm volatile("bar.sync 1, %0;" : : "r"(count));
}


/**
 * Floating point multiply. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FMUL_RZ(float a, float b)
{
	float d;
	asm("mul.rz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
	return d;
}


/**
 * Floating point multiply-add. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
{
	float d;
	asm("fma.rz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
	return d;
}


/**
 * Terminates the calling thread
 */
__device__ __forceinline__ void ThreadExit() {
	asm("exit;");
}	


/**
 * Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId()
{
	unsigned int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret) );
	return ret;
}



} // namespace cub
CUB_NS_POSTFIX
