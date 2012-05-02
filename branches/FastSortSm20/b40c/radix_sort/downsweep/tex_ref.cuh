/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * Texture references for downsweep kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>
#include <b40c/util/tex_vector.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Templated texture reference for downsweep keys (and values)
 */
template <
	typename KeyType,
	typename ValueType,
	int THREAD_ELEMENTS>
struct DownsweepTex
{
	enum {

		DEFAULT_KEY_TEX_VEC_SIZE  		= TexVector<KeyType, THREAD_ELEMENTS>::TEX_VEC_SIZE,
		DEFAULT_VALUE_TEX_VEC_SIZE 		= TexVector<ValueType, THREAD_ELEMENTS>::TEX_VEC_SIZE,
		TEX_VEC_SIZE 					= CUB_MIN(DEFAULT_KEY_TEX_VEC_SIZE, DEFAULT_VALUE_TEX_VEC_SIZE),
	};


	typedef typename TexVector<KeyType, TEX_VEC_SIZE>::TexVec 		KeyVectorType;		// Vector-type to use for textures
	typedef typename TexVector<KeyType, TEX_VEC_SIZE>::TexRef 		KeyTexRef;			// Texture type

	typedef typename TexVector<ValueType, TEX_VEC_SIZE>::TexVec 	ValueVectorType;	// Vector-type to use for textures
	typedef typename TexVector<ValueType, TEX_VEC_SIZE>::TexRef 	ValueTexRef;		// Texture type

	static KeyTexRef key_ref0;
	static KeyTexRef key_ref1;

	static ValueTexRef value_ref0;
	static ValueTexRef value_ref1;

	/**
	 * Bind textures
	 */
	static cudaError_t BindTexture(
		void *d_keys0,
		void *d_keys1,
		size_t key_bytes,
		void *d_values0,
		void *d_values1,
		size_t value_bytes)
	{
		cudaError_t retval = cudaSuccess;
		do {
			cudaChannelFormatDesc key_tex_desc = cudaCreateChannelDesc<KeyVectorType>();

			if (d_keys0) {
				// Bind key texture ref0
				retval = cudaBindTexture(0, key_ref0, d_keys0, key_tex_desc, bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
			if (d_keys1) {
				// Bind key texture ref1
				retval = cudaBindTexture(0, key_ref1, d_keys1, key_tex_desc, bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
			cudaChannelFormatDesc value_tex_desc = cudaCreateChannelDesc<ValueVectorType>();

			if (d_values0) {
				// Bind value texture ref0
				retval = cudaBindTexture(0, value_ref0, d_values0, value_tex_desc, value_bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
			if (d_values1) {
				// Bind value texture ref1
				retval = cudaBindTexture(0, value_ref1, d_values1, value_tex_desc, value_bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
		} while (0);

		return retval;
	}

};

// Texture reference definitions

template <typename KeyType, typename ValueType, int THREAD_ELEMENTS>
typename DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::KeyTexRef DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::key_ref0;

template <typename KeyType, typename ValueType, int THREAD_ELEMENTS>
typename DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::KeyTexRef DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::key_ref1;

template <typename KeyType, typename ValueType, int THREAD_ELEMENTS>
typename DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::ValueTexRef DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::value_ref0;

template <typename KeyType, typename ValueType, int THREAD_ELEMENTS>
typename DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::ValueTexRef DownsweepTex<KeyType, ValueType, THREAD_ELEMENTS>::value_ref1;




} // namespace downsweep
} // namespace radix_sort
} // namespace b40c

