
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main() {

	// Part B - omp Sparse
	int i, j;
	int K = 1000;
	double q = 0.15; // damping factor
	int NumPg = 1600;
	double b = 1.0 / NumPg;
	double s = 0.5;
	double startTime, endTime;

	// Create S Matrix
	// Number of non-zeros in matrix S
	int nnz = 2 * NumPg - 1;
	double *S = (double *)malloc(sizeof(double)*nnz);
	int *iS = (int *)malloc(sizeof(int)*NumPg + 1);
	int *jS = (int *)malloc(sizeof(int)*nnz);

#pragma omp parallel for
	for (i = 0; i < nnz; i++)
	{
		S[i] = 0.5;
	}
	S[2] = 1.0;
	
#pragma omp parallel for
	for (i = 0; i < NumPg; i++)
	{
		iS[i] = 2 * i;
	}
	iS[NumPg] = nnz;

#pragma omp parallel for
	for (i = 0; i < nnz; i++)
	{
		jS[i] = 0;
	}

	int counter = 0;
	for (i = 2; i < nnz; i += 2)
	{
		jS[i] = counter;
		counter += 1;
	}
	
	counter = 2;
	for (i = 3; i < nnz; i += 2)
	{
		jS[i] = counter;
		counter += 1;
	}

	jS[0] = 1;
	jS[1] = NumPg-1;

	// Create X vector
	double *X = (double *)malloc(sizeof(double)*NumPg);
	#pragma omp parallel for
	for (i = 0; i < NumPg; i++)
	{
		X[i] = 1.0 / NumPg;
	}

	// Start Timer
	startTime = omp_get_wtime();

	// Now find y = S*x
	int k,m;
	double *y = (double *)malloc(sizeof(double)*NumPg);

	for (m = 0; m < K; m++)
	{
		#pragma omp parallel for private(k)
		for (i = 0; i < NumPg; i++)
		{
			y[i] = 0.0;
			for (k = iS[i]; k < iS[i + 1]; k++)
			{
				y[i] += S[k] * X[jS[k]];
			}
		}

		// Now get y = Gx
		#pragma omp parallel for
		for (i = 0; i < NumPg; i++)
		{
			// implementing damping without extra storage
			y[i] = (1 - q)*y[i] + q / NumPg;
		}

		#pragma omp parallel for
		for (i = 0; i < NumPg; i++) {
			X[i] = y[i];
		}
	}

	// End Timer
	endTime = omp_get_wtime();

	// Find maximum value in Page Rank Vector
	double maximum;
	int location;
	int c;

	 maximum = X[0];

	  for (c = 1; c < NumPg; c++)
	  {
		if (X[c] > maximum)
		{
		   maximum  = X[c];
		   location = c+1;
		}
	  }

	printf("\nMaximum Value in Page Rank Vector is %.15f\n", maximum);

	// Find minimum value in Page Rank Vector
	double minimum;
	int location2;
	int d;

	minimum = X[0];

	for (d = 1; d < NumPg; d++)
	{
		if (X[d] < minimum)
		{
			minimum = X[d];
			location2 = d + 1;
		}
	}

	printf("\nMinimum Value in Page Rank Vector is %.15f\n", minimum);
	printf("\nRunTime: %.15f\n", endTime - startTime);

	free(iS);
	free(jS);
	free(S);
	free(X);
	free(y);

	return 0;
}
