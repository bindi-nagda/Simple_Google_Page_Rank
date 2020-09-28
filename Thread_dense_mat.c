
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


int main() {

	// Part A - omp dense
	int i, j;
	int K = 1000;
	double q = 0.15; // damping factor
	int NumPg = 1600;  // first do 16 then 1600
	double b = 1.0 / NumPg;
	double s = 0.5;
	double startTime, endTime;
	
	// Create G Matrix
	double *G = (double *)malloc(sizeof(double)*NumPg*NumPg);

#pragma omp parallel for private(j)
	for (i = 0; i < NumPg; i++)
	{
		for (j = 0; j < NumPg; j++)
		{
			if (abs(i - j) == 1)
			{
				G[i*NumPg + j] = (1 - q)*s + b * q;
			}
			else
			{
				G[i*NumPg + j] = 0.0 + b * q;
			}

		}
	}
	G[NumPg - 1] = (1 - q)*s + b * q;
	G[NumPg] = (1 - q)*1.0 + b * q;

	// Create X vector
	double *X = (double *)malloc(sizeof(double)*NumPg);
#pragma omp parallel for
	for (i = 0; i < NumPg; i++)
	{
		X[i] = 1.0 / NumPg;
	}

	// Start Timer
	startTime = omp_get_wtime();

	// Gx = y
	double *y = (double *)malloc(sizeof(double)*NumPg);
	int m;
	for (m = 0; m < K; m++)
	{
		#pragma omp parallel for private(j)
		for (i = 0; i < NumPg; i++) {
			y[i] = 0.0;
			for (j = 0; j < NumPg; j++) {
				y[i] += G[i*NumPg + j] * X[j];
			}
		}

		#pragma omp parallel for
		for (i = 0; i < NumPg; i++) {
			X[i] = y[i];
		}

	}

	// End Timer
	endTime = omp_get_wtime();

	// Find maximum value in Page Rank Vector
	double maximum = X[0];
	int location;
	int c;
	for (c = 1; c < NumPg; c++)
	{
		if (X[c] > maximum)
		{
			maximum = X[c];
			location = c + 1;
		}
	}

	printf("\nMaximum Value in Page Rank Vector is %.15f\n", maximum);
	
	// Find minimum value in Page Rank Vector
	double minimum = X[0];
	int location2;
	int d;
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
	
	free(X);
	free(y);
	free(G);

	return 0;
}
