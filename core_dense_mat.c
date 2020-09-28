#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"


int main(int argc, char *argv[]) {

	// Part C: MPI dense
	int i, j;
	int K = 1000;
	// damping factor
	double q = 0.15;
	int NumPg = 1600;
	double b = 1.0 / NumPg;
	double s = 0.5;
	double startTime, endTime;

	int numprocs,local_m,myid;
	
	/* Initialize MPI and get number of processes and my number or rank*/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	/* Processor zero sets the number of rows per processor*/
	if (myid == 0)
	{
		// Number of rows of the matrix per processor
		local_m = NumPg / numprocs;
		startTime = MPI_Wtime();
	}
	/* Broadcast number of rows for each processor to all processes */
	MPI_Bcast(&local_m, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
	// Create G Matrix
	double *local_G = (double *)malloc(sizeof(double)*local_m*NumPg);

	for (i = 0; i < local_m; i++)
	{
		for (j = 0; j < NumPg; j++)
		{
			if (abs((i+(local_m*myid))- j) == 1)
			{
				local_G[i*NumPg + j] = (1 - q)*s + b * q;
			}
			else
			{
				local_G[i*NumPg + j] = 0.0 + b * q;
			}
		}
	}

	if (myid == 0)
	{
		local_G[NumPg - 1] = (1 - q)*s + b * q;
		local_G[NumPg] = (1 - q)*1.0 + b * q;
	}

	// Create X vector
	double *local_X = (double *)malloc(sizeof(double)*local_m);
	for (i = 0; i < local_m; i++)
	{
		local_X[i] = 1.0 / NumPg;
	}

	// Gx = y
	double *local_y = (double *)malloc(sizeof(double)*local_m);
	double *global_X = (double *)malloc(sizeof(double)*NumPg);
	int m;

	//printf("\nThis is processor number %d\n", myid);

	// Start Timer
	if (myid == 0)
	{
		startTime = MPI_Wtime();
	}

	for (m = 0; m < K; m++)
	{
		MPI_Allgather(local_X, local_m, MPI_DOUBLE, global_X, local_m, MPI_DOUBLE, MPI_COMM_WORLD);
	
		for (i = 0; i < local_m; i++) 
		{
			local_y[i] = 0.0;
			for (j = 0; j < NumPg; j++) {
				local_y[i] += local_G[i*NumPg + j] * global_X[j];
			}
		}

		for (i = 0; i < local_m; i++) 
		{
			local_X[i] = local_y[i];
		}

	}

	double *global_y = (double *)malloc(sizeof(double)*NumPg);
	MPI_Gather(local_y, local_m, MPI_DOUBLE, global_y, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// End Timer
	if (myid == 0)
	{
		endTime = MPI_Wtime();
		printf("\nRunTime: %.15f\n", endTime - startTime);
	}

	if (myid == 0)
	{
		// Find maximum value in Page Rank Vector
		double maximum;
		int location;
		int c;
		maximum = global_y[0];

		for (c = 1; c < NumPg; c++)
		{
			if (global_y[c] > maximum)
			{
				maximum = global_y[c];
				location = c + 1;
			}
		}

		printf("\nMaximum Value in Page Rank Vector is %.15f\n", maximum);

		// Find minimum value in Page Rank Vector
		double minimum;
		int location2;
		int d;

		minimum = global_y[0];

		for (d = 1; d < NumPg; d++)
		{
			if (global_y[d] < minimum)
			{
				minimum = global_y[d];
				location2 = d + 1;
			}
		}

		printf("\nMinimum Value in Page Rank Vector is %.15f\n", minimum);
	}
	
	free(global_X);
	free(global_y);
	free(local_G);
	free(local_X);
	free(local_y);
	
	MPI_Finalize();

	return 0;
}
