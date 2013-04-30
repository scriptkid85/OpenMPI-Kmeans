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

float** mpiRead(string fname, int nline, int ndim, MPI_Comm  comm) {
	float data[nline][ndim], ** dataShard;
	const int root = 0;
	int rank, size;
	MPI_Status status;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_rank(comm, &size);

	/* everyone calls bcast, data is taken from root and ends up in everyone's buf */
	// send to other processes
	MPI_Bcast(&nline, 1, MPI_INT, 0, comm);
	MPI_Bcast(&ndim, 1, MPI_INT, 0, comm);

	// process id==0
	if (rank == root) {
		ifstream in(fname.c_str());
		string line, word;
		int i = 0;
		while (!in.eof()) {
			getline(in, line);
			stringstream ss(line.c_str());
			int j = 0;
			while( getline(ss, word, ',') ) {
				data[i][j] = atof(word.c_str());
				j++;
			}
			i++;
		}
		in.close();

		// send data to different processes
		int div = nline / size;	// # of lines for each block
		int rem = nline % size;
		int start = rem > 0 ? div+1 : div;
		int startInit = start;	// save for cooking rank 0's own data
		// start from the machine with rank 1 (rank 0 machine keep a part of data)
		for(int i = 1; i < size; i++) {
			// transfer data nearly equally
			int transSize = i > rem ? div+1 : div;
			// TODO: tag == 0?
			MPI_Send(data[start], ndim * transSize, MPI_FLOAT, i, 0, comm);
			start += transSize;
		}

		// cook rank 0's own data
		dataShard = new float[startInit][ndim];
		copy(&data[0][0], &data[startInit][0], dataShard);
		free(data);
	}
	// other processes
	else {
		int div = nline / size;	// # of lines for each block
		int rem = nline % size;
		nline = rank < rem ? div+1 : div;

		dataShard = new float[nline][ndim];
		MPI_Recv(dataShard, nline * ndim, MPI_FLOAT, root, rank, comm, &status);
	}

	return dataShard;
}

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
