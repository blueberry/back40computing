/**
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
 */


import java.util.*;

public class GenInput {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		Random random = new Random(System.currentTimeMillis());
		
		ArrayList<Integer> numbers = new ArrayList<Integer>();
		
		for (int i = 5; i <= 28; i++) {
			numbers.add(1 << i);
		}

		for (int i = 0; i < 1000; i++) {
			
			// pick float in [0.0, 1.0]
			double uniformSample = random.nextDouble();
			
			// scale to exp in [5.0, 28.0]
			double exp = ((28d - 5.0d) * uniformSample) + 5.0d;
			
			// output sample from log distribution of even numbers  [32.0, 2^28]
			int value = (int) Math.pow(2.0d, exp);

			numbers.add(value);
		}

		for (int i = 0; i < 2000; i++) {
			
			// pick int in [0.0, 2^28)
			int uniformSample = random.nextInt(268435456);
			int value = uniformSample + 32;

			numbers.add(value);
		}
		
		Collections.sort(numbers);
		
		// Dont print duplicates
		Integer prev = null;
		for (Integer num : numbers) {
			if ((prev != null) && (num.intValue() == prev.intValue())) {
				continue;
			}
			System.out.println(num);
			prev = num;
		}
	}

}

