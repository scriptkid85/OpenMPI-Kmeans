/*
 ============================================================================
 Name        : openMPI-Kmeans.c
 Author      : Guanyu Wang, Zeyuan Li
 Version     :
 Copyright   : 
 Description : Parallel K-means method using openMPI
 ============================================================================
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int kmean_write(char *filename_clustercenter, char *filename_belongtocluster,
		int numberofLocalData, int numberofTotalData, int numberofClusters,
		int numberofCoordinates, float **clusters, int *localMemebership,
		int ranktooutput, MPI_Comm comm) {
	MPI_File mpif;
	MPI_Status mpistatus;

	int i, j;
	int rank, nproc, err;
	char *str;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	if (rank == ranktooutput) {
		printf(
				"Start writing coordinates of K=%d cluster centers to file \"%s\"\n",
				numberofClusters, filename_clustercenter);

		err = MPI_File_open(comm, filename_clustercenter,
				MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpif);
		if (err != MPI_SUCCESS) {
			printf("Error: Cannot access file \"%s\"\n",
					filename_clustercenter);
			MPI_Finalize();
			exit(1);
		} else {
			for (i = 0; i < numberofClusters; i++) {
				char str[32];
				sprintf(str, "%d ", i);
				MPI_File_write(mpif, str, strlen(str), MPI_CHAR, &mpistatus);
				for (j = 0; j < numberofCoordinates; j++) {
					sprintf(str, "%f ", clusters[i][j]);
					MPI_File_write(mpif, str, strlen(str), MPI_CHAR,
							&mpistatus);
				}
				MPI_File_write(mpif, "\n", 1, MPI_CHAR, &mpistatus);
			}
		}
		MPI_File_close(&mpif);

		printf(
				"Start writing clusters to which all %d data belonging to file \"%s\"\n",
				numberofTotalData, filename_belongtocluster);
		err = MPI_File_open(comm, filename_belongtocluster,
				MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpif);
		if (err != MPI_SUCCESS) {
			printf("Error: Cannot access file \"%s\"\n",
					filename_belongtocluster);
			MPI_Finalize();
			exit(1);
		}

		//wait for the other memebership from other processors*/
		int divd = numberofTotalData / nproc;
		int rem = numberofTotalData % nproc;
		int outputCount = 0;
		int numberofRecData;
		for (i = 0; i < nproc; i++) {
			numberofRecData = (i < rem) ? divd + 1 : divd;
			if (i == ranktooutput) {
				//print out local membership
				for (j = 0; j < numberofLocalData; j++) {
					sprintf(str, "%d %d\n", outputCount++, localMemebership[j]);
					MPI_File_write(mpif, str, strlen(str), MPI_CHAR,
							&mpistatus);
				}
				continue;
			}
			MPI_Recv(localMemebership, numberofRecData, MPI_INT, i, i, comm,
					&mpistatus);
			//print out received membership
			for (j = 0; j < numberofRecData; j++) {
				sprintf(str, "%d %d\n", outputCount++, localMemebership[j]);
				MPI_File_write(mpif, str, strlen(str), MPI_CHAR, &mpistatus);
			}
		}
		MPI_File_close(&mpif);
	} else {
		MPI_Send(localMemebership, numberofLocalData, MPI_INT, ranktooutput,
				rank, comm);
	}
	return 1;
}

int main() {

}
