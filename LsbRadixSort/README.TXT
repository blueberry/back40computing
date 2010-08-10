
BUILD NOTES:
	
	The accompanying sorting benchtest should build under most Linux 
	platforms using CUDA Toolkit 3.0 or higher.

	NOTE: when adapting this code, you may consider compile the kernels to 
	use 32-bit device pointers.  Otherwise our kernel register counts 
	may be excessive, causing lmem spills and ~10% performance overhead.
	
	For more information, see our Google Code project site: 
	http://code.google.com/p/back40computing/

AUTHORS' REQUEST: 

	If you use|reference|benchmark this code, please cite our Technical 
	Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
	
	@TechReport{ Merrill:Sorting:2010,
       	author = "Duane Merrill and Andrew Grimshaw",
       	title = "Revisiting Sorting for GPGPU Stream Architectures",
       	year = "2010",
       	institution = "University of Virginia, Department of Computer Science",
       	address = "Charlottesville, VA, USA",
       	number = "CS2010-03"
	}
	
	Thanks!
