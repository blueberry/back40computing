/******************************************************************************
 * Copyright 2010 Duane Merrill
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
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <math.h>
#include <float.h>


namespace b40c {


/******************************************************************************
 * Templated routines for printing keys/values to the console 
 ******************************************************************************/

template<typename T> 
void PrintValue(T val) {
	printf("%d", val);
}

template<>
void PrintValue<float>(float val) {
	printf("%f", val);
}

template<>
void PrintValue<double>(double val) {
	printf("%f", val);
}

template<>
void PrintValue<unsigned char>(unsigned char val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned short>(unsigned short val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned int>(unsigned int val) {
	printf("%u", val);
}

template<>
void PrintValue<long>(long val) {
	printf("%ld", val);
}

template<>
void PrintValue<unsigned long>(unsigned long val) {
	printf("%lu", val);
}

template<>
void PrintValue<long long>(long long val) {
	printf("%lld", val);
}

template<>
void PrintValue<unsigned long long>(unsigned long long val) {
	printf("%llu", val);
}



/******************************************************************************
 * Helper routines for list construction and validation 
 ******************************************************************************/


/**
 * Generates random 32-bit keys.
 * 
 * We always take the second-order byte from rand() because the higher-order 
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 * 
 * We can decrease the entropy level of keys by adopting the technique 
 * of Thearling and Smith in which keys are computed from the bitwise AND of 
 * multiple random samples: 
 * 
 * entropy_reduction	| Effectively-unique bits per key
 * -----------------------------------------------------
 * -1					| 0
 * 0					| 32
 * 1					| 25.95
 * 2					| 17.41
 * 3					| 10.78
 * 4					| 6.42
 * ...					| ...
 * 
 */
template <typename K>
void RandomBits(K &key, int entropy_reduction = 0, int lower_key_bits = sizeof(K) * 8)
{
	const unsigned int NUM_USHORTS = (sizeof(K) + sizeof(unsigned short) - 1) / sizeof(unsigned short);
	unsigned short key_bits[NUM_USHORTS];
	
	do {
	
		for (int j = 0; j < NUM_USHORTS; j++) {
			unsigned short halfword = 0xffff; 
			for (int i = 0; i <= entropy_reduction; i++) {
				halfword &= (rand() >> 8);
			}
			key_bits[j] = halfword;
		}
		
		if (lower_key_bits < sizeof(K) * 8) {
			unsigned long long base = 0;
			memcpy(&base, key_bits, sizeof(K));
			base &= (1 << lower_key_bits) - 1;
			memcpy(key_bits, &base, sizeof(K));
		}
		
		memcpy(&key, key_bits, sizeof(K));
		
	} while (key != key);		// avoids NaNs when generating random floating point numbers 
}


/**
 * Compares the equivalence of two arrays
 */
template <typename T>
int CompareResults(T* computed, T* reference, const unsigned int len, bool verbose)
{
	for (int i = 0; i < len; i++) {

		if (computed[i] != reference[i]) {
			printf("Incorrect: [%d]: ", i);
			PrintValue<T>(computed[i]);
			printf(" != ");
			PrintValue<T>(reference[i]);

			if (verbose) {
				printf("\nresult[...");
				for (int j = -4; j <= 4; j++) {
					if ((i + j >= 0) && (i + j < len)) {
						PrintValue<T>(computed[i + j]);
						printf(", ");
					}
				}
				printf("...]");
				printf("\nreference[...");
				for (int j = -4; j <= 4; j++) {
					if ((i + j >= 0) && (i + j < len)) {
						PrintValue<T>(reference[i + j]);
						printf(", ");
					}
				}
				printf("...]");
			}

			return 1;
		}
	}

	printf("Correct");
	return 0;
}


}// namespace b40c
