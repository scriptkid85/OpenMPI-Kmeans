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
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "openMPI-Kmeans.h"
#define MY_MAXITER 1000

//
float** kmeans_read(char *fname, int *nline, int ndim, MPI_Comm comm) {
	float data[*nline][ndim], **dataShard;
	const int root = 0;
	int rank, size, i = 1;
	MPI_Status status;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_rank(comm, &size);

	/* everyone calls bcast, data is taken from root and ends up in everyone's buf */
	// send to other processes
	MPI_Bcast(nline, 1, MPI_INT, 0, comm);
	MPI_Bcast(&ndim, 1, MPI_INT, 0, comm);

	// process id==0
	if (rank == root) {
		FILE * fp;
		char * line = NULL;
		size_t len = 0;
		ssize_t read;

		fp = fopen(fname, "r");
		if (fp == NULL )
			exit(1);

		while ((read = getline(&line, &len, fp)) != -1) {
			printf("Retrieved line of length %zu :\n", read);
			printf("%s", line);
		}

		if (line)
			free(line);

		// send data to different processes
		int div = *nline / size;	// # of lines for each block
		int rem = *nline % size;
		int start = rem > 0 ? div + 1 : div;
		int startInit = start;	// save for cooking rank 0's own data
		// start from the machine with rank 1 (rank 0 machine keep a part of data)
		for (i = 1; i < size; i++) {
			// transfer data nearly equally
			int transSize = i > rem ? div + 1 : div;
			// TODO: tag == 0?
			MPI_Send(data[start], ndim * transSize, MPI_FLOAT, i, 0, comm);
			start += transSize;
		}

		// cook rank 0's own data
		dataShard = (float**) malloc(startInit * ndim * sizeof(float));
		copy(&data[0][0], &data[startInit][0], dataShard);
		free(data);
	}
	// other processes
	else {
		int div = *nline / size;	// # of lines for each block
		int rem = *nline % size;
		nline = rank < rem ? div + 1 : div;

		dataShard = (float**) malloc(*nline * ndim * sizeof(float));
		MPI_Recv(dataShard, *nline * ndim, MPI_FLOAT, root, rank, comm,
				&status);
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
	char str[1024];
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

//for find the nearest neighbor in the given set;
int find_NN(float *datapoint, float ** neighborset, int numberofNeighber,
		int numberofCoordinates) {
	int i, j;
	int nearest_neighbor = -1;
	float distance, mindist;
	mindist = FLT_MAX;
	for(i = 0; i < numberofNeighber; i++){
		distance = 0.0;
		for(j = 0; j < numberofCoordinates; j++){
			distance += (datapoint[j]-neighborset[i][j]) * (datapoint[j]-neighborset[i][j]);
			if(distance < mindist){
				nearest_neighbor = i;
			}
		}
	}
	return nearest_neighbor;
}

int kmeans(float **data, int numberofClusters, int numberofCoordinates,
		int numberofData, float stopthreshold, int *membership,
		float **clusters, MPI_Comm comm) {
	float **updatedClusters;
	int *updatedClusterSize;
	int *tmpClusterSize;

	int i, j;
	//initialization
	//malloc space for pointers
	updatedClusterSize = (int *) calloc(numberofClusters, sizeof(int));
	tmpClusterSize = (int *) calloc(numberofClusters, sizeof(int));

	updatedClusters = (float **) malloc(numberofClusters * sizeof(float*));
	updatedClusters[0] = (float *) calloc(numberofClusters * numberofCoordinates,
			sizeof(float));

	if (!updatedClusterSize || !tmpClusterSize || !updatedClusters
			|| !updatedClusters[0]) {
		printf("Error: Cannot calloc space for the new cluster variables");
		exit(1);
	}

	//reset memeber ship
	membership[0] = -1;
	for (i = 1; i < numberofData; i++) {
		updatedClusters[i] = updatedClusters[i - 1] + numberofCoordinates;
		membership[i] = -1;
	}

	//get the total data number
	int numberofTotalData;
	MPI_Allreduce(&numberofData, &numberofTotalData, 1, MPI_INT, MPI_SUM, comm);

	float delta;
	delta = FLT_MAX;
	int index, differences;
	int iterations;
	iterations = 0;
	while(delta > stopthreshold && iterations < MY_MAXITER){
		iterations ++;
//		may use the Wtime to record computing time
//		double time = MPI_Wtime();

		delta = 0.0;
		for(i = 0; i < numberofData; i++) {
			index = find_NN(data[i], clusters, numberofClusters, numberofCoordinates);
			if(index < 0){
				printf("Error: mistake in finding nearest cluster.");
				exit(1);
			}
			if(index != membership[i])differences ++;
			membership[i] = index;
			updatedClusterSize[index]++;
			for(j = 0; j < numberofCoordinates; j++) {
				updatedClusters[index][j] += data[i][j];
			}
		}

		//reduce all cluster's partial sums to the total sum for every cluster
		MPI_Allreduce(updatedClusters[0], clusters[0], numberofClusters * numberofCoordinates, MPI_FLOAT, MPI_SUM, comm);
		//reduce all cluster'size to get the total size for every cluster

		MPI_Allreduce(updatedClusterSize, tmpClusterSize, numberofClusters, MPI_INT, MPI_SUM, comm);

		//compute the new cluster center
		for(i = 0; i < numberofClusters; i++){
			for(j = 0; j < numberofCoordinates; j++){
				if(tmpClusterSize[i] > 0){
					clusters[i][j] /= tmpClusterSize[i];
				}
				updatedClusters[i][j] = 0;
			}
			updatedClusterSize[i] = 0;
		}

		int totaldifferences;
		MPI_Allreduce(&differences, &totaldifferences, 1, MPI_INT, MPI_SUM, comm);
		delta = totaldifferences / (double) numberofTotalData;
	}

	free(updatedClusters);
	free(updatedClusters[0]);
	free(updatedClusterSize);
	free(tmpClusterSize);
	return 1;
}
