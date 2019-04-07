#include <arm_neon.h>
#include <cblas.h>

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <memory.h>
#include <stdlib.h>
#include <sys/time.h>

// g++ -O3 sgemm.cpp -I ../OpenBLAS-0.2.20/ -o sgemm -L ../OpenBLAS-0.2.20 -lopenblas

void a64_sgemm_asimd_12x8(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, int bblocks, int K);
void sgemm_kernel_12x8(float *Apanel, float *Bpanel, float *Cpanel, int M, int N, int K);

//#include <stdlib.h>
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wB, unsigned int wA)
{
    for (unsigned int i = 0; i < wB; ++i)
        for (unsigned int j = 0; j < hA; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
				double a = A[k * hA + j];
				double b = B[i * wA + k];
				sum += a * b;
            }

            C[i * hA + j] = (float)sum;
        }
}

void packAs(float *src, float *dst, int kc, int lda)
{
	for (int i  = 0; i < kc; i++)
	{
		float *src_ptr = src + i*lda;
		*dst = *src_ptr;
		*(dst+1) = *(src_ptr + 1);
		*(dst+2) = *(src_ptr + 2);
		*(dst+3) = *(src_ptr + 3);
		*(dst+4) = *(src_ptr + 4);
		*(dst+5) = *(src_ptr + 5);
		*(dst+6) = *(src_ptr + 6);
		*(dst+7) = *(src_ptr + 7);

		dst += 8;
	}
}

void packBs(float *src, float *dst, int kc, int ldb)
{
	float *b_ptr_0 = &src[ 0];
	float *b_ptr_1 = b_ptr_0 + ldb;
	float *b_ptr_2 = b_ptr_1 + ldb;
	float *b_ptr_3 = b_ptr_2 + ldb;
	float *b_ptr_4 = b_ptr_3 + ldb;
	float *b_ptr_5 = b_ptr_4 + ldb;
	float *b_ptr_6 = b_ptr_5 + ldb;
	float *b_ptr_7 = b_ptr_6 + ldb;
	float *b_ptr_8 = b_ptr_7 + ldb;
	float *b_ptr_9 = b_ptr_8 + ldb;
	float *b_ptr_10 = b_ptr_9 + ldb;
	float *b_ptr_11 = b_ptr_10 + ldb;
	for (int i = 0; i < kc; i++)
	{
		*dst++ = *b_ptr_0++;
		*dst++ = *b_ptr_1++;
		*dst++ = *b_ptr_2++;
		*dst++ = *b_ptr_3++;
		*dst++ = *b_ptr_4++;
		*dst++ = *b_ptr_5++;
		*dst++ = *b_ptr_6++;
		*dst++ = *b_ptr_7++;
		*dst++ = *b_ptr_8++;
		*dst++ = *b_ptr_9++;
		*dst++ = *b_ptr_10++;
		*dst++ = *b_ptr_11++;
	}
}

void sgemm(float *A, float *B, float *C, int M, int N, int K)
{
	const int mc = 256;
	const int kc = 512;
	const int nc = 384;
	const int mr = 8;
	const int nr = 12;
	
	static float As[mc*kc]; // l2 cache
	static float Bs[kc*nc]; // l3 cache

	static float Cs[mr*nr]; // l1 cache

    memset(C, 0, M*N*sizeof(float));

	for (int j = 0; j < N; j += nc)
	{
		for (int k = 0; k < K; k += kc)
		{
			for (int i = 0; i < M; i += mc)
            {
				for (int jj = 0; jj < nc; jj += nr)
                {
					// pack a line
                    if (i == 0)
                        packBs(&B[k+(j+jj)*K], &Bs[jj*kc], kc, K);

					for (int ii = 0; ii < mc; ii += mr)
                    {
						// pack a line
                        if (jj == 0)
                            packAs(&A[i+ii+k*M], &As[ii*kc], kc, M);

                        sgemm_kernel_12x8(&As[ii*kc], &Bs[jj*kc], Cs, mr, nr, kc);

                        for (int n = 0; n < nr; n++)
                            for (int m = 0; m < mr; m++)
                                C[(j + jj + n)*M + i + ii + m] += Cs[n + m * nr];
					}
				}
			}
		}
    }
}

void sgemm_kernel_12x8(float *Apanel, float *Bpanel, float *Cpanel, int M, int N, int K)
{
	float *a_ptr = Apanel;
	float *b_ptr = Bpanel; 
	float *c_ptr = Cpanel;

	// double buffer
	int k = K / 2 - 1;

	// assembler
	asm volatile (
		// initialize C registers
		"movi v8.4s, #0x0\n"
		// load a0
		"ldr q0, [%[a_ptr]]\n"
		"movi v9.4s, #0x0\n"
		// load b0
		"ldr q2, [%[b_ptr]]\n"
		"movi v10.4s, #0x0\n"
		// load a1
		"ldr q1, [%[a_ptr], #16]\n"
		"movi v11.4s, #0x0\n"
		// load b1
		"ldr q3, [%[b_ptr], #16]\n"
		"movi v12.4s, #0x0\n"
		"prfm pldl1keep, [%[a_ptr], #64]\n"
		"movi v13.4s, #0x0\n"
		"prfm pldl1keep, [%[b_ptr], #64]\n"
		"movi v14.4s, #0x0\n"
		"prfm pldl1keep, [%[a_ptr], #128]\n"
		"movi v15.4s, #0x0\n"
		"prfm pldl1keep, [%[b_ptr], #128]\n"
		"movi v16.4s, #0x0\n"
		"prfm pldl1keep, [%[a_ptr], #192]\n"
		"movi v17.4s, #0x0\n"
		"prfm pldl1keep, [%[b_ptr], #192]\n"
		"movi v18.4s, #0x0\n"
		"prfm pldl1keep, [%[a_ptr], #256]\n"
		"movi v17.4s, #0x0\n"
		"prfm pldl1keep, [%[b_ptr], #256]\n"
		"movi v19.4s, #0x0\n"
		"prfm pldl1keep, [%[b_ptr], #320]\n"
		"movi v20.4s, #0x0\n"
		"prfm pldl1keep, [%[b_ptr], #384]\n"
		"movi v21.4s, #0x0\n"
		"movi v22.4s, #0x0\n"
		"movi v23.4s, #0x0\n"
		"movi v24.4s, #0x0\n"
		"movi v25.4s, #0x0\n"
		"movi v26.4s, #0x0\n"
		"movi v27.4s, #0x0\n"
		"movi v28.4s, #0x0\n"
		"movi v29.4s, #0x0\n"
		"movi v30.4s, #0x0\n"
		"movi v31.4s, #0x0\n"
		
		// if k euqal to zero, jump forwards (after) label 4
		"cbz %[k], 4f\n"

		// loop
		"1:\n"

		// ######################### unroll 1
		
		// b0 * a0
		"fmla v8.4s, v2.4s, v0.s[0]\n"
		"fmla v9.4s, v2.4s, v0.s[1]\n"
		"ldr q4, [%[b_ptr], #32]\n"	// load b2
		"fmla v10.4s, v2.4s, v0.s[2]\n"
		"prfm pldl1keep, [%[a_ptr], #320]\n"
		"fmla v11.4s, v2.4s, v0.s[3]\n"

		// b0 * a1
		"fmla v12.4s, v2.4s, v1.s[0]\n"
		"ldr q5, [%[a_ptr], #32]\n"	// load a0a
		"fmla v13.4s, v2.4s, v1.s[1]\n"
		"fmla v14.4s, v2.4s, v1.s[2]\n"
		"ldr q6, [%[a_ptr], #48]\n"	// load a1a
		"fmla v15.4s, v2.4s, v1.s[3]\n"
		"prfm pldl1keep, [%[b_ptr], #448]\n"

		// b1 * a0
		"fmla v16.4s, v3.4s, v0.s[0]\n"
		"ldr q2, [%[b_ptr], #48]\n"	// load b0 again
		"fmla v17.4s, v3.4s, v0.s[1]\n"
		"fmla v18.4s, v3.4s, v0.s[2]\n"
		"fmla v19.4s, v3.4s, v0.s[3]\n"

		// b1 * a1
		"fmla v20.4s, v3.4s, v1.s[0]\n"
		"prfm pldl1keep, [%[b_ptr], #512]\n"
		"fmla v21.4s, v3.4s, v1.s[1]\n"
		"fmla v22.4s, v3.4s, v1.s[2]\n"
		"fmla v23.4s, v3.4s, v1.s[3]\n"
		"ldr q3, [%[b_ptr], #64]\n"	// load b1 again

		// b2 * a0
		"fmla v24.4s, v4.4s, v0.s[0]\n"
		"fmla v25.4s, v4.4s, v0.s[1]\n"
		"fmla v26.4s, v4.4s, v0.s[2]\n"
		"fmla v27.4s, v4.4s, v0.s[3]\n"

		// b2 * a1
		"fmla v28.4s, v4.4s, v1.s[0]\n"
		"fmla v29.4s, v4.4s, v1.s[1]\n"
		"fmla v30.4s, v4.4s, v1.s[2]\n"
		"fmla v31.4s, v4.4s, v1.s[3]\n"
		"ldr q4, [%[b_ptr], #80]\n"	// load b2 again

		// ######################### unroll 2

		// b0 * a0a
		"fmla v8.4s, v2.4s, v5.s[0]\n"
		"fmla v9.4s, v2.4s, v5.s[1]\n"
		"fmla v10.4s, v2.4s, v5.s[2]\n"

		// b0 * a1a
		"fmla v11.4s, v2.4s, v5.s[3]\n"
		"fmla v12.4s, v2.4s, v6.s[0]\n"
		"ldr q0, [%[a_ptr], #64]\n"     // load a0
		"fmla v13.4s, v2.4s, v6.s[1]\n"
		"fmla v14.4s, v2.4s, v6.s[2]\n"
		"ldr q1, [%[a_ptr], #80]\n"     // load a1
		"fmla v15.4s, v2.4s, v6.s[3]\n"

		// b1 * a0a
		"fmla v16.4s, v3.4s, v5.s[0]\n"
		"ldr q2, [%[b_ptr], #96]\n"     // load b0 again
		"fmla v17.4s, v3.4s, v5.s[1]\n"
		"fmla v18.4s, v3.4s, v5.s[2]\n"
		"fmla v19.4s, v3.4s, v5.s[3]\n"

		// b1 * a1a
		"fmla v20.4s, v3.4s, v6.s[0]\n"
		"fmla v21.4s, v3.4s, v6.s[1]\n"
		"fmla v22.4s, v3.4s, v6.s[2]\n"
		"fmla v23.4s, v3.4s, v6.s[3]\n"
		"ldr q3, [%[b_ptr], #112]\n"     // load b1 again
		"add %[a_ptr], %[a_ptr], #64\n"	 // A += offsets
		"add %[b_ptr], %[b_ptr], #96\n"  // B += offsets

		// b2 * a0a
		"fmla v24.4s, v4.4s, v5.s[0]\n"
		"fmla v25.4s, v4.4s, v5.s[1]\n"
		"fmla v26.4s, v4.4s, v5.s[2]\n"
		"fmla v27.4s, v4.4s, v5.s[3]\n"
		"subs %[k], %[k], #1\n"		 // k = k - 1;

		// b2 * a1a
		"fmla v28.4s, v4.4s, v6.s[0]\n"
		"fmla v29.4s, v4.4s, v6.s[1]\n"
		"fmla v30.4s, v4.4s, v6.s[2]\n"
		"fmla v31.4s, v4.4s, v6.s[3]\n"
		"bne 1b\n"

		// ############################## out of loop
		"4:\n"
		// ######################### unroll 1

		// b0 * a0
		"fmla v8.4s, v2.4s, v0.s[0]\n"
		"fmla v9.4s, v2.4s, v0.s[1]\n"
		"ldr q4, [%[b_ptr], #32]\n"     // load b2
		"fmla v10.4s, v2.4s, v0.s[2]\n"
		"fmla v11.4s, v2.4s, v0.s[3]\n"

		// b0 * a1
		"fmla v12.4s, v2.4s, v1.s[0]\n"
		"ldr q5, [%[a_ptr], #32]\n"     // load a0a
		"fmla v13.4s, v2.4s, v1.s[1]\n"
		"fmla v14.4s, v2.4s, v1.s[2]\n"
		"ldr q6, [%[a_ptr], #48]\n"     // load a1a
		"fmla v15.4s, v2.4s, v1.s[3]\n"

		// b1 * a0
		"fmla v16.4s, v3.4s, v0.s[0]\n"
		"ldr q2, [%[b_ptr], #48]\n"     // load b0 again
		"fmla v17.4s, v3.4s, v0.s[1]\n"
		"fmla v18.4s, v3.4s, v0.s[2]\n"
		"fmla v19.4s, v3.4s, v0.s[3]\n"

		// b1 * a1
		"fmla v20.4s, v3.4s, v1.s[0]\n"
		"fmla v21.4s, v3.4s, v1.s[1]\n"
		"fmla v22.4s, v3.4s, v1.s[2]\n"
		"fmla v23.4s, v3.4s, v1.s[3]\n"
		"ldr q3, [%[b_ptr], #64]\n"     // load b1 again

		// b2 * a0
		"fmla v24.4s, v4.4s, v0.s[0]\n"
		"fmla v25.4s, v4.4s, v0.s[1]\n"
		"fmla v26.4s, v4.4s, v0.s[2]\n"
		"fmla v27.4s, v4.4s, v0.s[3]\n"

		// b2 * a1
		"fmla v28.4s, v4.4s, v1.s[0]\n"
		"fmla v29.4s, v4.4s, v1.s[1]\n"
		"fmla v30.4s, v4.4s, v1.s[2]\n"
		"fmla v31.4s, v4.4s, v1.s[3]\n"
		"ldr q4, [%[b_ptr], #80]\n"     // load b2 again

		// ######################### unroll 2
		"add %[a_ptr], %[a_ptr], #64\n"  // A += offsets
		"add %[b_ptr], %[b_ptr], #96\n"  // B += offsets
		"fmla v8.4s, v2.4s, v5.s[0]\n"
		"fmla v16.4s, v3.4s, v5.s[0]\n"
		"fmla v9.4s, v2.4s, v5.s[1]\n"
		"str q8, [%[c_ptr]]\n"
		"fmla v24.4s, v4.4s, v5.s[0]\n"
		"str q16, [%[c_ptr], #16]\n"
		"fmla v17.4s, v3.4s, v5.s[1]\n"
		"str q24, [%[c_ptr], #32]\n"
		"fmla v25.4s, v4.4s, v5.s[1]\n"
		"str q9, [%[c_ptr], #48]\n"
		"fmla v10.4s, v2.4s, v5.s[2]\n"
		"str q17, [%[c_ptr], #64]\n"
		"fmla v18.4s, v3.4s, v5.s[2]\n"
		"str q25, [%[c_ptr], #80]\n"
		"fmla v26.4s, v4.4s, v5.s[2]\n"
		"str q10, [%[c_ptr], #96]\n"
		"fmla v11.4s, v2.4s, v5.s[3]\n"
		"str q18, [%[c_ptr], #112]\n"
		"fmla v19.4s, v3.4s, v5.s[3]\n"
		"str q26, [%[c_ptr], #128]\n"
		"fmla v27.4s, v4.4s, v5.s[3]\n"
		"str q11, [%[c_ptr], #144]\n"
		"fmla v12.4s, v2.4s, v6.s[0]\n"
		"str q19, [%[c_ptr], #160]\n"
		"fmla v20.4s, v3.4s, v6.s[0]\n"
		"str q27, [%[c_ptr], #176]\n"
		"fmla v28.4s, v4.4s, v6.s[0]\n"
		"str q12, [%[c_ptr], #192]\n"
		"fmla v13.4s, v2.4s, v6.s[1]\n"
		"str q20, [%[c_ptr], #208]\n"
		"fmla v21.4s, v3.4s, v6.s[1]\n"
		"str q28, [%[c_ptr], #224]\n"
		"fmla v29.4s, v4.4s, v6.s[1]\n"
		"str q13, [%[c_ptr], #240]\n"
		"fmla v14.4s, v2.4s, v6.s[2]\n"
		"str q21, [%[c_ptr], #256]\n"
		"fmla v22.4s, v3.4s, v6.s[2]\n"
		"str q29, [%[c_ptr], #272]\n"
		"fmla v30.4s, v4.4s, v6.s[2]\n"
		"str q14, [%[c_ptr], #288]\n"
		"fmla v15.4s, v2.4s, v6.s[3]\n"
		"str q22, [%[c_ptr], #304]\n"
		"fmla v23.4s, v3.4s, v6.s[3]\n"
		"str q30, [%[c_ptr], #320]\n"
		"fmla v31.4s, v4.4s, v6.s[3]\n"
		"str q15, [%[c_ptr], #336]\n"
		"str q23, [%[c_ptr], #352]\n"
		"str q31, [%[c_ptr], #368]\n"
		"add %[c_ptr], %[c_ptr], #384\n"

		: [k] "+r" (k),
		  [a_ptr] "+r" (a_ptr),
		  [b_ptr] "+r" (b_ptr),
		  [c_ptr] "+r" (c_ptr)
		: "0" (k),
		  "1" (a_ptr),
		  "2" (b_ptr),
		  "3" (c_ptr)
		: "v0", "v1", "v2", "v3",
		  "v4", "v5", "v6", "v7",
		  "v8", "v9", "v10", "v11",
		  "v12", "v13", "v14", "v15",
		  "v16", "v17", "v18", "19",
		  "v20", "v21", "v22", "v23",
		  "cc", "memory"
	);
}

#define M (64*8)
#define N (64*12)
#define K 1024
int main(int argc, char** argv)
{
	// parse input
	int nIter = 100;
	if (argc == 2)
		nIter = atoi(argv[1]);

    openblas_set_num_threads(1);

	// variables
    float *A, *B, *C, *Cblas, *C1;
	A = new float[M * K];
	B = new float[K * N];
	C = new float[M * N];
    Cblas = new float[M * N];
	C1 = new float[M * N];

	// random init
	for (int i = 0; i < M * K; i++)
		A[i] = (float)rand() / RAND_MAX;
	for (int i = 0; i < K * N; i++)
        B[i] = (float)rand() / RAND_MAX;
	
    // 12x8 kernel
	struct timeval start, stop;


    // 12x8 kernel
    {
        gettimeofday(&start, NULL);
        // execute for loop
        for (int i = 0; i < nIter; i++) {
            sgemm(A, B, C, M, N, K);
        }
        gettimeofday(&stop, NULL);
        double t = (double)(stop.tv_sec-start.tv_sec) + (stop.tv_usec-start.tv_usec) * 1e-6; // sec
        t /= nIter;	// average time
        double gflps = 2.0 * M * N * K * 1e-9 / t; // gflops
        printf("12x8 kernel matrix(%d,%d,%d), flops: %fGFLOPS, time %f sec!\n", M, N, K, gflps, t);
    }

    // cblas
    {
        gettimeofday(&start, NULL);
        // execute for loop
        for (int i = 0; i < nIter; i++) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, M, B, K, 0.0, Cblas, M);
        }
        gettimeofday(&stop, NULL);
        double t = (double)(stop.tv_sec-start.tv_sec) + (stop.tv_usec-start.tv_usec) * 1e-6; // sec
        t /= nIter;	// average time
        double gflps = 2.0 * M * N * K * 1e-9 / t; // gflops
        printf("cblas matrix(%d,%d,%d), flops: %fGFLOPS, time %f sec!\n", M, N, K, gflps, t);
    }

	// element-wise
	matrixMulCPU(C1, A, B, M, N, K);
	for(int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
            float diff = fabs(C1[i*M+j] - Cblas[i*M+j]);
			if (diff > 1e-3)
			{
				printf("ERROR: i,j %d,%d, diff is %f, C %f, C1 %f\n", 
                        i, j, diff, Cblas[i*M+j], C1[i*M+j]);
                return 0;
			}
		}
	}
    printf("Test Passed!\n");

	delete []A;
	delete []B;
	delete []C;
    delete []Cblas;
    delete []C1;
	return 0;
}
