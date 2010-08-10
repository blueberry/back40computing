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


import java.io.*;
import java.util.*;

public class ParseCudaProf
{
	
/*
	String 	method;
	float 	gputime;
	float 	cputime;
	float 	occupancy;
	int		gld_uncoalesced;
	int 	gld_coalesced; 
	int 	gld_request; 
	int 	gld_32b;
	int 	gld_64b;
	int 	gld_128b;
	int 	gst_uncoalesced;
	int 	gst_coalesced;
	int 	gst_request;
	int 	gst_32b;
	int 	gst_64b;
	int 	gst_128b;
	int 	local_load;
	int 	local_store;
	int 	branch;
	int 	divergent_branch;
	int 	instructions;
	int 	warp_serialize;
	int		cta_launched;
*/	
	
	// --cudpp flag
	public static boolean cudpp;
	
	
	public static class DoneException extends Exception {
		private static final long serialVersionUID = 1L;
	}
	
	
	public abstract static class Accumulator {
		
		public String name;
		public int iterations = 0;

		static public Accumulator Create(String name) {
			if (name.equals("gputime") || name.equals("cputime") || name.equals("occupancy")) {
				return new DoubleAccumulator(name);
			}

			return new IntAccumulator(name);
		}
		
		public abstract void Accumulate(String token);
		public abstract double Average();
		public abstract double Total();
	}

	public static class IntAccumulator extends Accumulator {
		
		public int accumulator;
		
		public IntAccumulator(String name) {
			this.name = name;
			accumulator = 0;
		}
		
		public void Accumulate(String token) {
			accumulator += Integer.parseInt(token);
			iterations++;
		}
		
		public double Average() {
			return ((double) accumulator) / iterations;
		}

		public double Total() {
			return (double) accumulator;
		}
	}
	
	public static class DoubleAccumulator extends Accumulator {
		
		public double accumulator;
		
		public DoubleAccumulator(String name) {
			this.name = name;
			accumulator = 0f;
		}
		
		public DoubleAccumulator(String name, double initvalue) {
			this.name = name;
			accumulator = initvalue;
		}

		public void Accumulate(String token) {
			accumulator += Double.parseDouble(token);
			iterations++;
		}
		
		public double Average() {
			return accumulator / iterations;
		}

		public double Total() {
			return (double) accumulator;
		}
	}


	public static class KernelData {

		public String name;
		
		public ArrayList<Accumulator> accumulators = new ArrayList<Accumulator>();
		
		public KernelData(String name, ArrayList<String> columns_lines) {
			this.name = name;
			
			for (String columns_line : columns_lines) {
				StringTokenizer st = new StringTokenizer(columns_line, ",");
				
				// skip "method"
				st.nextToken();

				// add accumulators for each column
				while (st.hasMoreTokens()) {
					String counter = st.nextToken();
					accumulators.add(Accumulator.Create(counter));
				}
			}
		}
		
		public void Accumulate(ArrayList<String> counters_lines) throws Exception {

			int index = 0;
			
			for (String counters_line : counters_lines) {
				StringTokenizer st = new StringTokenizer(counters_line, ",");
				
				// skip "method"
				st.nextToken();

				// accumulate each value
				while (st.hasMoreTokens()) {
					String value = st.nextToken();
					accumulators.get(index).Accumulate(value);
					index++;
				}
			}
		}
		
		public Accumulator GetCounter(String name) {
			for (Accumulator counter : accumulators) {
				if (counter.name.equals(name)) {
					return counter;
				}
			}
			return null;
		}
		
		public void AddCounter(Accumulator counter) {
			accumulators.add(counter);
		}
	}
	
	public static void Usage() {
		System.out.println("Parse [--cudpp] <iterations> <profile files...>");
	}
	
	public static String patterns[] = { 

			// SRTS sorting
			"Reduc", 
			"SrtsScanSpine", 
			"FlushKernel", 
			"SrtsScanDigitBulk",
			"SrtsSingleCtaSort",
		
			// CUDPP sorting
			"radixSortSingleWarp",
			"radixSortBlocks",
			"findRadixOffsets",
			"scan4",
			"vectorAddUniform4",
			"reorderData",
			"emptyKernel"
	};
	
	public static void DisplayProblem(
			HashMap<String, KernelData> kernels, 
			int iterations,
			String specific_kernels[]
	) {

		if (kernels.values().isEmpty()) {
			return;
		}

		double gputime = 0;
		
		double total_instrs = 0;
		double warp_serialize = 0;
		double total_gld = 0;
		double total_gst = 0;

		for (KernelData kernel : kernels.values()) {
			Accumulator gputime_ac = kernel.GetCounter("gputime");
			Accumulator instrs_ac = kernel.GetCounter("instructions");
			Accumulator warpser_ac = kernel.GetCounter("warp_serialize");
			Accumulator gld_32b_ac = kernel.GetCounter("gld_32b");
			Accumulator gld_64b_ac = kernel.GetCounter("gld_64b");
			Accumulator gld_128b_ac = kernel.GetCounter("gld_128b");
			Accumulator gst_32b_ac = kernel.GetCounter("gst_32b");
			Accumulator gst_64b_ac = kernel.GetCounter("gst_64b");
			Accumulator gst_128b_ac = kernel.GetCounter("gst_128b");

			double kernel_gld = 0.0;
			double kernel_gst = 0.0;
			
			if (gputime_ac != null) {
				gputime += gputime_ac.Total();
			}
			
			if (warpser_ac != null) {
				warp_serialize += warpser_ac.Total();
			}
			
			if (instrs_ac != null) {
				total_instrs += instrs_ac.Total();
			}

			if (gld_32b_ac != null) {
				kernel_gld += (gld_32b_ac.Total() * 32);
				total_gld += (gld_32b_ac.Total() * 32);
			}

			if (gld_64b_ac != null) {
				kernel_gld += (gld_64b_ac.Total() * 64);
				total_gld += (gld_64b_ac.Total() * 64);
			}

			if (gld_128b_ac != null) {
				kernel_gld += (gld_128b_ac.Total() * 128);
				total_gld += (gld_128b_ac.Total() * 128);
			}

			if (gst_32b_ac != null) {
				kernel_gst += (gst_32b_ac.Total() * 32);
				total_gst += (gst_32b_ac.Total() * 32);
			}

			if (gst_64b_ac != null) {
				kernel_gst += (gst_64b_ac.Total() * 64);
				total_gst += (gst_64b_ac.Total() * 64);
			}

			if (gst_128b_ac != null) {
				kernel_gst += (gst_128b_ac.Total() * 128);
				total_gst += (gst_128b_ac.Total() * 128);
			}
	
			kernel.AddCounter(new DoubleAccumulator("gld", kernel_gld));
			kernel.AddCounter(new DoubleAccumulator("gst", kernel_gst));
		}

		System.out.print(gputime / iterations);
		System.out.print(", " + ((warp_serialize / iterations) * 30));
		System.out.print(", " + ((total_instrs / iterations) * 32.0 * 30.0));
		System.out.print(", " + ((total_gld / iterations) * 10.0));
		System.out.print(", " + ((total_gst / iterations) * 10.0));

		for (String name : specific_kernels) {
			KernelData kernel = kernels.get(name);
			if (kernel != null) {
				System.out.print(", " + ((kernel.GetCounter("gputime") == null) ? 0 : (kernel.GetCounter("gputime").Total() / iterations)));
				System.out.print(", " + ((kernel.GetCounter("warp_serialize") == null) ? 0 : (kernel.GetCounter("warp_serialize").Total() / iterations) * 30.0));
				System.out.print(", " + ((kernel.GetCounter("instructions") == null) ? 0 : (kernel.GetCounter("instructions").Total() / iterations) * 32.0 * 30.0));
				System.out.print(", " + ((kernel.GetCounter("gld") == null) ? 0 : (kernel.GetCounter("gld").Total() / iterations) * 10.0));
				System.out.print(", " + ((kernel.GetCounter("gst") == null) ? 0 : (kernel.GetCounter("gst").Total() / iterations) * 10.0));
			} else {
				System.out.print(", 0.0");
				System.out.print(", 0.0");
				System.out.print(", 0.0");
				System.out.print(", 0.0");
				System.out.print(", 0.0");
			}
		}
		
		System.out.println();
	}
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception
	{
		if (args.length < 2) {
			Usage();
			return; 
		}
		
		int nextarg = 0;
		if (args[nextarg].indexOf("cudpp") != -1) {
			nextarg++;
			cudpp = true;
		}
		
		// Kernels we're interested in printing specific data for
		String[] specific_kernels;
		if (cudpp) {
			// CUDPP sorting
			String[] kernels = { 
					"radixSortBlocks",
					"findRadixOffsets",
					"reorderData"};
			specific_kernels = kernels;
			
		} else {

			// SRTS sorting
			String[] kernels = { 
					"Reduc",
					"SrtsScanDigitBulk"};
			specific_kernels = kernels;
		}

		System.out.print("gputime");
		System.out.print(", warp_serialize");
		System.out.print(", total_instrs");
		System.out.print(", total_gld");
		System.out.print(", total_gst");


		for (String name : specific_kernels) {
			System.out.print(", " + name + "_gputime");
			System.out.print(", " + name + "_serialize");
			System.out.print(", " + name + "_instrs");
			System.out.print(", " + name + "_gld");
			System.out.print(", " + name + "_gst");
		}

		System.out.println();

		int iterations = Integer.parseInt(args[nextarg]);
		nextarg++;

		// Initialize readers
		ArrayList<BufferedReader> readers = new ArrayList<BufferedReader>();
		for (int i = nextarg; i < args.length; i++) {
			readers.add(new BufferedReader(new FileReader(args[i])));
		}
		
		// Determine columns
		ArrayList<String> columns_lines = new ArrayList<String>();
		for (BufferedReader reader : readers) {
			String line;
			while (true) {
				line = reader.readLine();				
				if ((line.indexOf("NV_Warning") < 0) &&
					(line.indexOf("# CUDA_") < 0) &&
					(line.indexOf("# TIMESTAMPFACTOR") < 0))
				{
					break;
				}
			}
			columns_lines.add(line);
		}
		
		HashSet<String> unmatched = new HashSet<String>();
		
		try {

			// Loop over different problem instances
			while (true) {
			
				HashMap<String, KernelData> kernels = new HashMap<String, KernelData>();

				int itr = 0;
				boolean matched = false;
				
				// Loop over iterations within the same problem instance 
				while (itr <= iterations) {

					ArrayList<String> counters_lines = new ArrayList<String>();

					// read list of lines
					for (BufferedReader reader : readers) {
						String line = reader.readLine();
						if (line == null) {
							DisplayProblem(kernels, iterations, specific_kernels);
							throw new DoneException();
						}
						counters_lines.add(line);
					}

					// retrieve kernel data for line(s)
					KernelData kernel = null;
					for (String pattern : patterns) {
						if (counters_lines.get(0).indexOf(pattern) != -1) {
							kernel = kernels.get(pattern);
							if (kernel == null) {
								kernel = new KernelData(pattern, columns_lines);
								kernels.put(pattern, kernel);
							}
						}
					}
					
					if (kernel == null) {
						// didn't match pattern
						
						if ((counters_lines.get(0).indexOf("memcopy") > -1) || (counters_lines.get(0).indexOf("memcpy") > -1)) {
							// ignore memcopies altogether
							continue;
						}

						unmatched.add(counters_lines.get(0).substring(0, 5));
						if (matched) {
							// we were matched in, and now we're not
							DisplayProblem(kernels, iterations, specific_kernels);
							break;
						}
						continue;
					}
					
					if (!matched) {
						// transitioned into a new iteration
						itr++;
						matched = true;
					}
				
					// accumulate counters 
					kernel.Accumulate(counters_lines);
				}

			}
			
		} catch (DoneException e) {}

		System.out.println();
	}
	
}
