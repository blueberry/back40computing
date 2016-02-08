
---

## DEPRECATED! ##
The B40C project has been superseded by the [CUB library](http://nvlabs.github.io/cub/).  CUB borrows and improves upon many of the ideas and algorithms we developed for the B40C and [MGPU](http://nvlabs.github.io/moderngpu/) projects.

<br><br>
<hr />
<h2>OVERVIEW</h2>
This project is a collection of fast, efficient GPU primitives.<br>
<br>
The goal of this work is to push the "speed of light" for GPU hardware, and then attempt to distill what we learned, namely:<br>
<ul><li>Reusable design patterns (problem decomposition, software composition)<br>
</li><li>Reusable code (stream primitives, kernel sub-primitives, etc.)</li></ul>

<i>"Back Forty".  The Back Forty is a colloquialism for the furthest 40-acre parcel of land on a farm.  In particular, the nickname was given to particular Hollywood studio backlot owned by RKO/Desilu Pictures that was used to construct epic movie sets such as King Kong.  The Back Forty is where your big work gets done.</i>

<br><br>
<hr />
<h2>SUB-PROJECTS</h2>
<b><i><a href='RadixSorting.md'>SRTS Radix Sorting</a></i></b>
<ul><li>High performance GPU sorting, implemented in CUDA</li></ul>

<blockquote>Merrill, D. and Grimshaw, A. <a href='https://sites.google.com/site/duanemerrill/PplGpuSortingPreprint.pdf'>High Performance and Scalable Radix Sorting: A case study of implementing dynamic parallelism for GPU computing</a>. Parallel Processing Letters, vol. 21, no. 2, 2011, pp. 245-272.</blockquote>

<br>
<br>
<BR><br>
<br>
<br>
<br>
<b><i>GPU Graph Traversal</i></b>
<ul><li>High performance, work-optimal BFS traversal of sparse graphs, implemented in CUDA.</li></ul>

<blockquote>Merrill, D., Garland, M., and Grimshaw, A. <a href='https://sites.google.com/site/duanemerrill/ppo213s-merrill.pdf'>Scalable GPU Graph Traversal</a>.  In Proceedings of the 17th ACM SIGPLAN symposium on Principles and Practice of Parallel Programming (PPoPP '12).  ACM, New York, NY, USA, pp. 117-128.</blockquote>

<br><br>
<hr />
<h2>Contributors</h2>
<ul><li><a href='http://www.cs.virginia.edu/~dgm4d/'>Duane Merrill</a>