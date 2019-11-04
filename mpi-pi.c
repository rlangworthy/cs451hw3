#include <mpi.h>
#include <math.h>
#include <stdio.h>

int main(argc,argv)
int argc;
char *argv[];
{
    int done = 0, n, myid, numprocs, i, maxrank;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    maxrank=numprocs-1;
    while (!done)
    {
	if (myid == 0) {
	    printf("Enter the number of intervals: (0 quits) ");
	    scanf("%d",&n);
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (n == 0) break;
  
	h   = 1.0 / (double) n;
	sum = 0.0;
	for (i = myid + 1; i <= n; i += numprocs) {
	    x = h * ((double)i - 0.5);
	    sum += 4.0 / (1.0 + x*x);
	}
	mypi = h * sum;
    
	MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, maxrank,
		   MPI_COMM_WORLD);
    
	if (myid == maxrank)
	    printf("pi is approximately %.16f, Error is %.16f\n",
		   pi, fabs(pi - PI25DT));
    MPI_Barrier(MPI_COMM_WORLD)
    }
    MPI_Finalize();
    return 0;
}
