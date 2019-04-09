armv8_sgemm: A single precision GEMM example on armv8 using assembly code.
===

Introduction:
---

This is not a fully GEMM library, just a simple exaple of GEMM. 
The dimensions only support (M,N,K) times (64*8,64*12,256).
The example only supports C=A*B. And we write this project after the study of arm ComputeLibrary gemm kernel(12x8).
Matrices A,B,C are column-major format. And we donnot use multi-thread.

***

Pre-requisites:
---

	A C++ compiler (tested with GCC)
	OpenBLAS (tested with version 0.2.20)
***

Test:
---

	Dimension(M,N,K): (512, 768, 1024)
	OpenBLAS: 21 GFLOPS
	OURS: 25 GFLOPS
***

Kernel(12x8):
===
Register Allocation:
---

	ARMv8 has 32 128bit floating-point registers labeled v0-v31.
	According to gotoBLAS paper, the inner loop is a (mrxnr) GESS kernel. (mr,nr) is single precision register number.
	So how to decide register blocking factor (mrxnr), we donot describe the details.
	The critical point is to maximize the compute-to-memory access ratio under some constraints(eg. total 32 register).
	Suppose A registers factor mr=8 (2 128bit), B registers factor nr=12 (3 128bit), so C registers mr*nr=96 (24 128bit).
	So we need 29(2+3+24) 128bit registers at least. And we left 3 128bit registers.
	The ARM ComputeLibrary use 2 registers to double A_next.
***

Register Chart:
---

	A : v0, v1
	A': v5, v6
	B : v2, v3, v4
	C : v8 ~ v31
	Ignore: v7 
                                                B
        	        |        v2        |        v3        |        v4        |
	     
        	|    |  |        v8        |        v16       |        v24       |
        	|    |  |        v9        |        v17       |        v25       |
        	| v0 |  |        v10       |        v18       |        v26       |
        	|    |  |        v11       |        v19       |        v27       |
              A0
        	|    |  |        v12       |        v20       |        v28       |
        	|    |  |        v13       |        v21       |        v29       |
        	| v1 |  |        v14       |        v22       |        v30       |
        	|    |  |        v15       |        v23       |        v31       |
***

Loop Unroll: 
---

	unroll 0:
                                             B
        	        |        v2        |          v3      |        v4        |
	     
        	|    |  | fmla v2, v0.s[0] | fmla v3, v0.s[0] | fmla v4, v0.s[0] |
        	|    |  | fmla v2, v0.s[1] | fmla v3, v0.s[1] | fmla v4, v0.s[1] |
        	| v0 |  | fmla v2, v0.s[2] | fmla v3, v0.s[2] | fmla v4, v0.s[2] |
        	|    |  | fmla v2, v0.s[3] | fmla v3, v0.s[3] | fmla v4, v0.s[3] |
              A0
        	|    |  | fmla v2, v1.s[0] | fmla v3, v1.s[0] | fmla v4, v1.s[0] |
        	|    |  | fmla v2, v1.s[1] | fmla v3, v1.s[1] | fmla v4, v1.s[1] |
        	| v1 |  | fmla v2, v1.s[2] | fmla v3, v1.s[2] | fmla v4, v1.s[2] |
        	|    |  | fmla v2, v1.s[3] | fmla v3, v1.s[3] | fmla v4, v1.s[3] |
        	
	unroll 1:    	
                                             B
        	        |        v2        |          v3      |        v4        |
	     
        	|    |  | fmla v2, v5.s[0] | fmla v3, v5.s[0] | fmla v4, v5.s[0] |
        	|    |  | fmla v2, v5.s[1] | fmla v3, v5.s[1] | fmla v4, v5.s[1] |
        	| v5 |  | fmla v2, v5.s[2] | fmla v3, v5.s[2] | fmla v4, v5.s[2] |
        	|    |  | fmla v2, v5.s[3] | fmla v3, v5.s[3] | fmla v4, v5.s[3] |
              A1
        	|    |  | fmla v2, v6.s[0] | fmla v3, v6.s[0] | fmla v4, v6.s[0] |
        	|    |  | fmla v2, v6.s[1] | fmla v3, v6.s[1] | fmla v4, v6.s[1] |
        	| v6 |  | fmla v2, v6.s[2] | fmla v3, v6.s[2] | fmla v4, v6.s[2] |
        	|    |  | fmla v2, v6.s[3] | fmla v3, v6.s[3] | fmla v4, v6.s[3] |
***

Improvement Analysis:
---
Register Rotation, we can use register v7 to get more loop unrolling.
I only found a unroll factor 4 closed the circle. 
Other solutions have more unrollings, but I didn't find the close circle.

Unroll 4 solution as follows, more details see the images in my project. But we didnot finish this solution:

	A0: v0, v1
	A1: v5, v6
	B0: v2, v3, v4
	B1: v7, v2, v3
	B2: v4, v7, v2
	B3: v3, v4, v7
	C : v8 ~ v31


	unroll 0:
                                             B0
        	        |        v2        |          v3      |        v4        |
	     
        	|    |  | fmla v2, v0.s[0] | fmla v3, v0.s[0] | fmla v4, v0.s[0] |
        	|    |  | fmla v2, v0.s[1] | fmla v3, v0.s[1] | fmla v4, v0.s[1] |
        	| v0 |  | fmla v2, v0.s[2] | fmla v3, v0.s[2] | fmla v4, v0.s[2] |
        	|    |  | fmla v2, v0.s[3] | fmla v3, v0.s[3] | fmla v4, v0.s[3] |
              A0
        	|    |  | fmla v2, v1.s[0] | fmla v3, v1.s[0] | fmla v4, v1.s[0] |
        	|    |  | fmla v2, v1.s[1] | fmla v3, v1.s[1] | fmla v4, v1.s[1] |
        	| v1 |  | fmla v2, v1.s[2] | fmla v3, v1.s[2] | fmla v4, v1.s[2] |
        	|    |  | fmla v2, v1.s[3] | fmla v3, v1.s[3] | fmla v4, v1.s[3] |
        	
	unroll 1:    	
                                             B1
        	        |        v7        |          v2      |        v3        |
	     
        	|    |  | fmla v7, v5.s[0] | fmla v2, v5.s[0] | fmla v3, v5.s[0] |
        	|    |  | fmla v7, v5.s[1] | fmla v2, v5.s[1] | fmla v3, v5.s[1] |
        	| v5 |  | fmla v7, v5.s[2] | fmla v2, v5.s[2] | fmla v3, v5.s[2] |
        	|    |  | fmla v7, v5.s[3] | fmla v2, v5.s[3] | fmla v3, v5.s[3] |
              A1
        	|    |  | fmla v7, v6.s[0] | fmla v2, v6.s[0] | fmla v3, v6.s[0] |
        	|    |  | fmla v7, v6.s[1] | fmla v2, v6.s[1] | fmla v3, v6.s[1] |
        	| v6 |  | fmla v7, v6.s[2] | fmla v2, v6.s[2] | fmla v3, v6.s[2] |
        	|    |  | fmla v7, v6.s[3] | fmla v2, v6.s[3] | fmla v3, v6.s[3] |


	unroll 2:
                                             B2
        	        |        v4        |          v7      |        v2        |
	     
        	|    |  | fmla v4, v0.s[0] | fmla v7, v0.s[0] | fmla v2, v0.s[0] |
        	|    |  | fmla v4, v0.s[1] | fmla v7, v0.s[1] | fmla v2, v0.s[1] |
        	| v0 |  | fmla v4, v0.s[2] | fmla v7, v0.s[2] | fmla v2, v0.s[2] |
        	|    |  | fmla v4, v0.s[3] | fmla v7, v0.s[3] | fmla v2, v0.s[3] |
              A0
        	|    |  | fmla v4, v1.s[0] | fmla v7, v1.s[0] | fmla v2, v1.s[0] |
        	|    |  | fmla v4, v1.s[1] | fmla v7, v1.s[1] | fmla v2, v1.s[1] |
        	| v1 |  | fmla v4, v1.s[2] | fmla v7, v1.s[2] | fmla v2, v1.s[2] |
        	|    |  | fmla v4, v1.s[3] | fmla v7, v1.s[3] | fmla v2, v1.s[3] |
        	
	unroll 3:    	
                                             B3
        	        |        v3        |          v4      |        v7        |
	     
        	|    |  | fmla v3, v5.s[0] | fmla v4, v5.s[0] | fmla v7, v5.s[0] |
        	|    |  | fmla v3, v5.s[1] | fmla v4, v5.s[1] | fmla v7, v5.s[1] |
        	| v5 |  | fmla v3, v5.s[2] | fmla v4, v5.s[2] | fmla v7, v5.s[2] |
        	|    |  | fmla v3, v5.s[3] | fmla v4, v5.s[3] | fmla v7, v5.s[3] |
              A1
        	|    |  | fmla v3, v6.s[0] | fmla v4, v6.s[0] | fmla v7, v6.s[0] |
        	|    |  | fmla v3, v6.s[1] | fmla v4, v6.s[1] | fmla v7, v6.s[1] |
        	| v6 |  | fmla v3, v6.s[2] | fmla v4, v6.s[2] | fmla v7, v6.s[2] |
        	|    |  | fmla v3, v6.s[3] | fmla v4, v6.s[3] | fmla v7, v6.s[3] |
***
Reference:
---
	http://blog.chinaunix.net/uid-20706279-id-1888741.html
	https://blog.csdn.net/wuyao721/article/details/3573598#_Toc200964859
	http://www.keil.com/support/man/docs/armclang_ref/armclang_ref_qjl1517569411293.htm


	http://apfel.mathematik.uni-ulm.de/~lehn/FLENS-Trinity/flens/examples/tut01-page08.html
	http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/
