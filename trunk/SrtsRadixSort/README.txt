
BUILD NOTES:
	
	The accompanying sorting benchtest should build under most Linux 
	platforms using CUDA Toolkit 3.0 or higher.

	NOTE: when adapting this code, you should compile the kernels to 
	use 32-bit device pointers.  Otherwise our kernel register counts 
	will be excessive and we won't be able to meet our targeted 
	threadblock occupancy.
	
	For more information, see our Google Code project site: 
	http://code.google.com/p/back40computing/

AUTHORS' REQUEST: 

	If you use|reference|benchmark this code, please cite our Technical 
	Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
	
	Duane Merrill and Andrew Grimshaw, "Revisiting Sorting for GPGPU Stream 
	Architectures," University of Virginia, Department of Computer Science, 
	Charlottesville, VA, USA, Technical Report CS2010-03, 2010.
	
	Thanks!
