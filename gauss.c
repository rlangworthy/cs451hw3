/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c" 
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 3000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN +1], X[MAXN], R[MAXN][MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

void verify() {
  float sol;
  int i;
  for(i = 0; i < N; i++){
    sol += A[0][i] * X[i];
  }
  printf("A[0]*X = %5.2f\nB[0] = %5.2f\n", sol, B[0]);
}

int main(int argc, char **argv) {
  /* Timing variables */
    /* Elapsed times using times() */
  int myid, numprocs;
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */
  double etstart, etstop;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  if(myid==0){
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

    etstart = MPI_Wtime();
  }
  /* Start Clock */

  /* Gaussian Elimination */
  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;
    /* Elapsed times using gettimeofday() */

  if(myid == 0) {
      printf("Computing.\n");
  }

  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  /* Gaussian elimination */
  for(norm = 0; norm < N - 1; norm++) {
    int acounts[numprocs],bcounts[numprocs], adispl[numprocs], bdispl[numprocs];
    int i, offset, div;
    div = (N-(norm+1))/numprocs;
    offset = 0;
    for(i = 0;i < numprocs;i++){
        bdispl[i] = offset;
        bcounts[i] = div;
        offset += bcounts[i];
        acounts[i] = bcounts[i] * MAXN;
        adispl[i] = bdispl[i] * MAXN;
    }
    bcounts[numprocs-1] += ((N-(norm+1))%numprocs);
    acounts[numprocs-1] = bcounts[numprocs-1] * MAXN;
    if(myid == 0){
        B[MAXN] = A[norm][norm];
    }
    if(myid==0){
        printf("B on norm %i\n", norm);
        for (col = 0; col < N; col++) {
            printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
        }
        printf("A[norm][norm] %f\n", B[MAXN]);
    }
    MPI_Bcast(&B, MAXN+1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(myid==0){
        MPI_Scatterv(&A[norm+1][0], acounts, adispl, MPI_FLOAT, MPI_IN_PLACE, acounts[myid], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else {
        MPI_Scatterv(&A[norm+1][0], acounts, adispl, MPI_FLOAT, &A[norm + 1 + bdispl[myid]][0], acounts[myid], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    for (row = norm + 1 + bdispl[myid]; row < norm + 1 + bdispl[myid] + bcounts[myid]; row ++) {
        multiplier = A[row][norm] / B[MAXN];
        for (col = norm; col < N; col++) {
            A[row][col] -= A[norm][col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;
    }

    if(myid==0){
        MPI_Gatherv(MPI_IN_PLACE, acounts[myid], MPI_FLOAT, &A[norm+1][0], acounts, adispl, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(MPI_IN_PLACE, bcounts[myid], MPI_FLOAT, &B[norm+1], bcounts, bdispl, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gatherv(&A[norm + 1 + bdispl[myid]][0], acounts[myid], MPI_FLOAT, &A[norm+1][0], acounts, adispl, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&B[norm + 1 + bdispl[myid]], bcounts[myid], MPI_FLOAT, &B[norm+1], bcounts, bdispl, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
  }

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  if(myid==0){
    /* Back substitution */
    for (row = N - 1; row >= 0; row--) {
        X[row] = B[row];
        for (col = N-1; col > row; col--) {
        X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
  }

  /* Stop Clock */

  if(myid==0){
    etstop = MPI_Wtime();
    printf("Time Elapsed: %1.2f\n", etstop-etstart);
    /* Display output */
    print_X();

    verify();

    /* Display timing results 
    printf("\nElapsed time = %g ms.\n",
        (float)(usecstop - usecstart)/(float)1000);

    printf("(CPU times are accurate to the nearest %g ms)\n",
        1.0/(float)CLOCKS_PER_SEC * 1000.0);
    printf("My total CPU time for parent = %g ms.\n",
        (float)( (cputstop.tms_utime + cputstop.tms_stime) -
            (cputstart.tms_utime + cputstart.tms_stime) ) /
        (float)CLOCKS_PER_SEC * 1000);
    printf("My system CPU time for parent = %g ms.\n",
        (float)(cputstop.tms_stime - cputstart.tms_stime) /
        (float)CLOCKS_PER_SEC * 1000);
    printf("My total CPU time for child processes = %g ms.\n",
        (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
            (cputstart.tms_cutime + cputstart.tms_cstime) ) /
        (float)CLOCKS_PER_SEC * 1000);
    Contrary to the man pages, this appears not to include the parent */
    printf("--------------------------------------------\n");}
  MPI_Finalize();
  exit(0);
}
