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
 * 
 * 
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
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <b40c_util.h>

/**
 * Verifies the "less than" property for sorted lists 
 */
template <typename T>
int VerifySort(T* sorted_keys, const unsigned int len, bool verbose) 
{
	for (int i = 1; i < len; i++) {

		if (sorted_keys[i] < sorted_keys[i - 1]) {
			printf("Incorrect: [%d]: ", i);
			b40c::PrintValue<T>(sorted_keys[i]);
			printf(" < ");
			b40c::PrintValue<T>(sorted_keys[i - 1]);

			if (verbose) {	
				printf("\n\n[...");
				for (int j = -4; j <= 4; j++) {
					if ((i + j >= 0) && (i + j < len)) {
						b40c::PrintValue<T>(sorted_keys[i + j]);
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
