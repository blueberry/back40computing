## Sorting performance as a function of effective key bits ##
_(and why real-world sorting rates may be higher than reported here)_





&lt;BR&gt;



---


### 

&lt;FONT color=#006666&gt;

The importance of write-coalescing in sorting passes

&lt;/FONT&gt;

 ###

Achieving good write-locality is important for minimizing the number of memory transactions, or _coalesced writes_.  NVIDIA GPUs obtain maximum bandwidth when they are able to coalesce concurrent memory accesses: for a SIMD instruction that accesses global memory, the individual accesses for each thread within the warp can be combined/bundled together by the memory subsystem into a single memory transaction if every reference falls within the same contiguous global memory segment.

For each sorting pass, the appropriate digit from each key is extracted and used to identify the corresponding bin (of the radix-_r_ bins) that key is to be scattered into.  The write-offsets into those bins are computed via prefix scans. Once it has these write-offsets, each scattering thread could use this information to write its keys directly into the the output bins.  However, doing so would result in poor write coherence: a random distribution of key-bits would have threads within the same warp writing keys to many different bins (i.e., poor write-locality).

Instead we use the local prefix sums to scatter them to local shared memory where consecutive threads can pick up consecutive keys and then scatter them to global device memory with a minimal number of memory transactions. We want most warps to be writing consecutive keys to only one or two distinct bins (i.e., good write-locality).

The distribution of key-bits can have an impact on the overall sorting performance. Memory bandwidth is wasted when a warp's threads scatter keys to more than one bin: the memory subsystem pushes a full-sized transaction through to DRAM (e.g., 32B / 64B / 128B), even though only a portion of it contains actual data.

When compiled for radix-_r_ = 2 bins, our key-scattering writes for uniformly-random key distributions incur a ~70% I/O overhead (i.e., 70% of write bandwidth is wasted by partially-full transactions).  This overhead increases in proportion with _r_: it is 92% in the current implementation (where _r_ = 16 bins).



&lt;BR&gt;



---


### 

&lt;FONT color=#006666&gt;

The impact of key-entropy on sorting performance

&lt;/FONT&gt;

 ###

The expectation is that a perfectly uniform distribution of key bits will yield the largest number of fragmented scatter transactions, and therefore the worst performance.  As bits become less uniform, the expected number of bins to be scattered into decreases (as certain digits become more likely than others).

We confirm this by using a technique for generating key sequences with different _entropy levels_ [1](1.md).  The idea is to generate keys that trend towards having more 0s than 1s bits by repeatedly bitwise-ANDing together uniformly-random bits.  For each bitwise-AND, the number of 1s bits per key decreases.  (An infinite number of bitwise-AND iterations results in a completely uniform key-distribution of all 0s.)

| _**Entropy-reduction iterations**_	| _**Effectively-unique bits per key**_ |
|:-----------------------------------|:--------------------------------------|
| 0					                             | 32                                    |
| 1					                             | 25.95                                 |
| 2					                             | 17.41                                 |
| 3					                             | 10.78                                 |
| 4					                             | 6.42                                  |
| ...					                           | ...                                   |
| oo					                            | 0                                     |


As shown below, our scatter inefficiencies decrease for less-random key distributions.  With zero-effective random bits (uniformly identical keys), our key-value sorting rates (_d_= 4-bits, _r_ = 16 bins) improve by 1.13x on GTX-480, and by 1.12x on GTX-285.


> http://back40computing.googlecode.com/svn/wiki/images/sorting/entropyreduction.PNG


```
  [1] K. Thearling and S. Smith, “An improved supercomputer sorting benchmark,” in Proc. ACM/IEEE Conference on Supercomputing, 1992, pp. 14–19.
```