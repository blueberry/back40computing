<table><tr><td>
<img src='http://back40computing.googlecode.com/svn/wiki/images/SortingSmall.jpg' />
</td><td valign='top'>
<b><br>
<blockquote></b><h1>Table of Contents</h1>
<b></blockquote></b>

<blockquote>
</td></tr></table></blockquote>



&lt;BR&gt;



---


# News #

  * We are now also distributing an alternative sorting enactor designed specifically for small problem sizes that are not large enough to saturate the GPU (e.g., problems < 1.2M elements.). It performs multiple digit-place passes over the input problem all within a single kernel launch. It performs up to 3x faster than our original, high-bandwidth sorting implementation for such problems.

  * Our sorting algorithms have been incorporated into the [Thrust Library of Parallel Primitives](http://code.google.com/p/thrust/), v.1.3.0.

  * For discussion, see our [CUDA Forum thread](http://forums.nvidia.com/index.php?showtopic=175238)





&lt;BR&gt;



---


# Sorting Overview #

This project implements a very fast, efficient radix sorting method for CUDA-capable devices.  For sorting large sequences of fixed-length keys (and values), we believe our sorting primitive to be the fastest available for any fully-programmable microarchitecture: our stock NVIDIA GTX480 sorting results **exceed the 1G keys/sec** average sorting rate (i.e., one billion 32-bit keys sorted per second).

In addition, one of our design goals for this project is flexibility.  We've designed our implementation to adapt itself and perform well on all generations and configurations of programmable NVIDIA GPUs, and for a wide variety of input types.

**Features:**

  * The current implementation can sort:
    * Keys of any C/C++ built-in, numeric type (e.g., `signed char`, `float`, `unsigned long long`, etc.)
    * Any structured payload type (within reason).

  * Early-exit digit-passes.  When the sorting operation detects that all keys have the same digit at the same digit-place, the pass for that digit-place is short-circuited, reducing the cost of that pass by 80%.  This makes our implementation suitable for even low-degree binning problems where sorting would normally be overkill. (We note that the random keys used to produce our performance results do not trigger this feature.)

  * Explicit passes for small problems.  Our small-problem, single-grid sorting enactor allows the caller to specify the lower-order bits over which the keys should be sorted (e.g., the lower 17 bits out of 32-bit keys).  This reduces the number of overall sorting passes (4-bits per pass) for scenarios in which the keyspace can be restricted in this manner.

[Download](#Building_and_Adapting.md) and play around with the implementation yourself!


#### 

&lt;FONT color=#006666&gt;

Radix Sorting

&lt;/FONT&gt;

 ####

The radix sorting method works by iterating over the _d_-bit digit-places of the keys from least-significant to most-significant.  For each digit-place, the method performs a stable distribution sort of the keys based upon their digit at that digit-place.  Given an _n_-element sequence of _k_-bit keys and a radix _r_ = 2_<sup>d</sup>_, a radix sort of these keys will require _k_/_d_ iterations of a distribution sort over all n keys.

The distribution sort (a.k.a. counting sort) is the fundamental component of the radix sorting method.  In a data-parallel, shared-memory model of computation, each logical processor gathers its key, decodes the specific digit at the given digit-place, and then must cooperate with other processors to determine where the key should be relocated.  The relocation offset will be the key's global rank, i.e., the number of keys with lower digits at that digit place plus the number of keys having the same digit, yet occurring earlier in the input sequence.

We implement these distribution sorting passes using a very efficient implementation of a generalized parallel prefix scan.  Our generalized scan is designed to operate over multiple, concurrent scan problems.  For example, with _d_ = 4 bits (_r_ = 16 digits), our multi-scan does sixteen scan operations: one for each digit.  For each of the scan operations (e.g., the 0s scan, the 1s scan, the 2s scan, etc.), the input for each key is a 1 if the key's digit place contains that operation's digit, 0 otherwise.  When the mulit-scan is done, the logical processor for each key can look up the scan result from the appropriate scan operation to determine where its key should be placed.

#### 

&lt;FONT color=#006666&gt;

Authors' Request

&lt;/FONT&gt;

 ####

If you use/reference/benchmark this code, please cite our [PPL Journal Article](http://back40computing.googlecode.com/svn/wiki/documents/PplGpuSortingPreprint.pdf):

_D. Merrill and A. Grimshaw, “High Performance and Scalable Radix Sorting: A case study of implementing dynamic parallelism for GPU computing,” Parallel Processing Letters, vol. 21, no. 2, pp. 245-272, 2011._

Bibtex:

```
 @article{
	title = {High Performance and Scalable Radix Sorting: A case study of implementing dynamic parallelism for {GPU} computing},
	volume = {21},
	issn = {0129-6264},
	shorttitle = {High Performance and Scalable Radix Sorting},
	url = {http://www.worldscinet.com/ppl/21/2102/S0129626411000187.html},
	doi = {10.1142/S0129626411000187},
	number = {02},
	journal = {Parallel Processing Letters},
	author = {Duane Merrill and Andrew Grimshaw},
	year = {2011},
	pages = {245--272}
 }
```



&lt;BR&gt;



&lt;BR&gt;



---


# Performance #

The following figures and tables present sorting rates for various CUDA GPUs.  These results were measured using a suite of ~3,000 randomly-sized input sequences (sized 32 - 272M elements), each initialized with keys and values whose bits were sampled from a uniformly random distribution.  For a discussion of why these results represent a lower-bound on sorting performance (i.e., real-world sorting rates may be higher), see our evaluation of [key entropy](KeyEntropyReduction.md).

Our measurements for elapsed time were taken directly from GPU hardware performance counters.  Therefore these results are reflective of in situ sorting problems: they preclude the driver overhead and the overheads of staging data to/from the accelerator, allowing us to directly contrast the individual and cumulative performance of the stream kernels involved.

Please note that the following measurements were made using the Cuda 3.0 compiler and driver framework.  (In our evaluation, code generated by the 3.0 compiler is ~0.5% faster than that generated by the 3.1 compiler.)  For Fermi-class devices, we compiled our device kernels for 32-bit device pointers.  (See the section on [building and adapting](#Building_and_Adapting.md) below).

We invite the reader to compare our results with the current state-of-the-art for GPUs and CPUs (and for experimental architectures), e.g.:
  * GPU sorting implementations (e.g., [CUDPP](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5161005))
  * CPU implementations (e.g., 240M 32-bit keys/sec for the [for the Intel quad-core Core i7](http://doi.acm.org/10.1145/1807167.1807207))
  * Experimental many-core implementations (e.g., 386M - 560M 32-bit keys/sec for [for the Intel Larrabee](http://doi.acm.org/10.1145/1454159.1454171) and [for the Intel MIC/Knight's Ferry](http://techresearch.intel.com/spaw2/uploads/files/FASTsort_CPUsGPUs_IntelMICarchitectures.pdf))


### 

&lt;FONT color=#006666&gt;

Selected Performance Plots

&lt;/FONT&gt;

 ###


> Sorting rates as a function of problem size for 32-bit `uint` keys:

> http://back40computing.googlecode.com/svn/wiki/images/sorting/32bitkeys.PNG

> 

&lt;BR&gt;



> And for 32-bit `uint` keys paired with 32-bit values:

> http://back40computing.googlecode.com/svn/wiki/images/sorting/32bitpairs.PNG


### 

&lt;FONT color=#006666&gt;

Average Saturated Sorting Rates

&lt;/FONT&gt;

 ###

Unless otherwise noted, the following tables present average saturated sorting rates for problem sizes greater than or equal to 32M unsigned keys.  Our implementation is also capable of sorting signed and floating-point key types (e.g., `float`, `int`, `double`, etc.) at a  0.5% - 1.5% performance overhead.



&lt;BR&gt;


| 

&lt;FONT color=#006666&gt;

**8 Bit Keys**

&lt;/FONT&gt;

 |
|:--------------------------------------------------------------|
| <i>10<sup>6</sup> elements / sec</i>                          |                                                               | **C2050 (No ECC)**                                            | **C2050 (ECC)**                                               | **C1060**                                                     |                                                               | **GTX 480**                                                   | **GTX 460 (1GB)**                                             | **GTX 285**                                                   | **GTX 280**                                                   | **9800 GTX+**                                                 | **8800 GTX**|
| <b>Keys-only</b>                                              |                                                               | 3,188.67                                                      | 2,599.39                                                      | 1,849.95                                                      |                                                               | 4,310.29                                                      |                                                               | 2,108.10                                                      | 1,831.18                                                      | 310.07                                                        |                                                               |
| <b>32-bit values</b>                                          |                                                               | 2,465.85                                                      | 1,774.62                                                      | 1,620.92                                                      |                                                               | 3,391.34                                                      |                                                               | 1,852.49                                                      | 1,612.23                                                      | 273.62                                                        |                                                               |
| <b>64-bit values</b>                                          |                                                               | 1,904.50                                                      | 1,136.93                                                      | 1,068.16                                                      |                                                               | 2,416.20                                                      |                                                               | 1,526.05                                                      | 1,394.84                                                      | 254.46<sup>�</sup>                                            |                                                               |
| <b>128-bit values</b>                                         |                                                               | 731.30                                                        | 613.10                                                        |   295.44                                                      |                                                               | 888.38                                                        |                                                               | 418.77<sup>�</sup>                                            | 373.60<sup>�</sup>                                            | 154.54<sup>��</sup>                                           |                                                               |



&lt;BR&gt;


| 

&lt;FONT color=#006666&gt;

**16 Bit Keys**

&lt;/FONT&gt;

 |
|:---------------------------------------------------------------|
| <i>10<sup>6</sup> elements / sec</i>                           |                                                                | **C2050 (No ECC)**                                             | **C2050 (ECC)**                                                | **C1060**                                                      |                                                                | **GTX 480**                                                    | **GTX 460 (1GB)**                                              | **GTX 285**                                                    | **GTX 280**                                                    | **9800 GTX+**                                                  | **8800 GTX**|
| <b>Keys-only</b>                                               |                                                                | 1,572.03                                                       | 1,270.92                                                       | 989.32                                                         |                                                                | 2,129.14                                                       |                                                                | 1,128.11                                                       | 979.19                                                         | 184.93                                                         |                                                                |
| <b>32-bit values</b>                                           |                                                                | 1,209.04                                                       | 811.15                                                         | 801.33                                                         |                                                                | 1,656.37                                                       |                                                                | 970.28                                                         | 845.22                                                         | 162.88                                                         |                                                                |
| <b>64-bit values</b>                                           |                                                                | 958.20                                                         | 518.56                                                         | 516.15                                                         |                                                                | 1,139.69                                                       |                                                                | 743.45                                                         | 676.64                                                         | 150.68<sup>�</sup>                                             |                                                                |
| <b>128-bit values</b>                                          |                                                                | 361.50                                                         | 293.92                                                         | 147.90                                                         |                                                                | 436.78                                                         |                                                                | 210.81<sup>�</sup>                                             | 187.41<sup>�</sup>                                             | 81.59<sup>��</sup>                                             |                                                                |



&lt;BR&gt;


| 

&lt;FONT color=#006666&gt;

**32 Bit Keys**

&lt;/FONT&gt;

 |
|:---------------------------------------------------------------|
| <i>10<sup>6</sup> elements / sec</i>                           |                                                                | **C2050 (No ECC)**                                             | **C2050 (ECC)**                                                | **C1060**                                                      |                                                                | **GTX 480**                                                    | **GTX 460 (1GB)**                                              | **GTX 285**                                                    | **GTX 280**                                                    | **9800 GTX+**                                                  | **8800 GTX**|
| <b>Keys-only</b>                                               |                                                                | 741.52                                                         | 580.41                                                         | 523.84                                                         |                                                                | 1,005.49                                                       |                                                                | 615.36                                                         | 534.53                                                         | 265.19                                                         |                                                                |
| <b>32-bit values</b>                                           |                                                                | 581.14                                                         | 350.01                                                         | 343.00                                                         |                                                                | 775.12                                                         |                                                                | 489.72                                                         | 449.46                                                         | 188.71<sup>�</sup>                                             |                                                                |
| <b>64-bit values</b>                                           |                                                                | 423.85                                                         | 221.88                                                         | 234.40                                                         |                                                                | 492.28                                                         |                                                                | 346.98                                                         | 315.10                                                         | 169.73<sup>�</sup>                                             |                                                                |
| <b>128-bit values</b>                                          |                                                                | 172.65                                                         | 134.56                                                         | 71.60                                                          |                                                                | 206.16                                                         |                                                                | 102.98<sup>�</sup>                                             | 92.06<sup>�</sup>                                              | 54.86<sup>��</sup>                                             |                                                                |



&lt;BR&gt;


| 

&lt;FONT color=#006666&gt;

**64 Bit Keys**

&lt;/FONT&gt;

 |
|:---------------------------------------------------------------|
| <i>10<sup>6</sup> elements / sec</i>                           |                                                                | **C2050 (No ECC)**                                             | **C2050 (ECC)**                                                | **C1060**                                                      |                                                                | **GTX 480**                                                    | **GTX 460 (1GB)**                                              | **GTX 285**                                                    | **GTX 280**                                                    | **9800 GTX+**                                                  | **8800 GTX**|
| <b>Keys-only</b>                                               |                                                                | 240.3                                                          | 161.1                                                          | 170.20                                                         |                                                                | 304.86                                                         |                                                                | 219.55                                                         | 191.18                                                         | 70.21<sup>�</sup>                                              |                                                                |
| <b>32-bit values</b>                                           |                                                                | 190.9                                                          | 105.8                                                          | 108.23                                                         |                                                                | 224.30                                                         |                                                                | 157.80                                                         | 143.01                                                         | 54.86<sup>�</sup>                                              |                                                                |
| <b>64-bit values</b>                                           |                                                                | 157.6                                                          | 91.01                                                          | 94.09                                                          |                                                                | 186.40                                                         |                                                                | 138.59<sup>�</sup>                                             | 125.70<sup>�</sup>                                             | 51.20<sup>��</sup>                                             |                                                                |
| <b>128-bit values</b>                                          |                                                                | 76.08                                                          | 59.41                                                          | 33.12                                                          |                                                                | 90.61<sup>�</sup>                                              |                                                                | 47.51<sup>�</sup>                                              | 42.46<sup>�</sup>                                              | 24.23<sup>��</sup>                                             |                                                                |

<sup>�</sup> 16M+ elements (restricted by global memory size)

&lt;BR&gt;


<sup>��</sup> 8M+ elements (restricted by global memory size)





&lt;BR&gt;



&lt;BR&gt;



---


# Building and Adapting #

This subproject can be found in the `FastSortSm20` branch.  (An older, slightly slower version exists in the main branch.) It contains C++/CUDA code for:

  * The GPU kernel sources
  * A templated API for invoking a sorting operation from a host program.  By including the sorting API like a header file, `nvcc` will specialize kernels and runtime code needed to sort the data types specified by the enclosing application.
  * A simple toy program (and a more advanced benchtest program) that demonstrate how the sorting implementation can be integrated into and called from host programs.

Like [Thrust](http://code.google.com/p/thrust/), the sources for our implementation are designed to be included and built as part of larger applications (as opposed to linked in as a shared or static library).  This prevents kernel bloat and allows the implementation to tailor itself to your data types.

The subproject sources require compilation using CUDA Toolkit version 3.0 or higher.  The repository includes a `Makefile` to build the example programs on Linux.  This Makefile can be used as a guide for constructing an equivalent VC++ project for evaluation on Windows systems.

When performance benchmarking our implementation, we suggest that you compile the kernels with the following options:

  * 32-bit device pointers, regardless of whether we're on a 64-bit machine or not.  64-bit device pointers incur ~10-15% slowdown because it causes fairly increased kernel register counts which can prevent us from being able to meet our targeted occupancies.





&lt;BR&gt;



&lt;BR&gt;



---


Cheers!