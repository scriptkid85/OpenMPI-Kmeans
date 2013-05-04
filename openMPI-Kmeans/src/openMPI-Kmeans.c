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
#include <math.h>
#define MY_MAXITER 500

//
float** kmeans_read(char *fname, int *nline, int ndim, MPI_Comm comm) {
	float data[*nline][ndim], **dataShard;
	const int root = 0;
	int rank, size, i = 1, num = 0;
	char *token;
	MPI_Status status;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	//printf("rank:%d\tsize:%d\n", rank, size);

	/* everyone calls bcast, data is taken from root and ends up in everyone's buf */
	// send to other processes
//	MPI_Bcast(nline, 1, MPI_INT, 0, comm);
//	MPI_Bcast(&ndim, 1, MPI_INT, 0, comm);

	//printf("nline:%d\tndim:%d\n", *nline, ndim);

	// process id==0
	if (rank == root) {
		FILE * fp;
		char * line = NULL;
		size_t len = 0;
		ssize_t read;

		fp = fopen(fname, "r");
		if (fp == NULL )
			exit(EXIT_FAILURE);

		while ((read = getline(&line, &len, fp)) != -1) {
			int j = 0;
			token = strtok(line, " ,");
			while (token != NULL ) {
				data[num][j++] = atof(token);
//				printf("%s\t", token);
				token = strtok(NULL, " ,");
			}
			num++;
//			printf("%d\n", num);
//			printf("Retrieved line of length %zu :\n", read);
//			printf("%s", line);
		}

		// send data to different processes
		int div = *nline / size;	// # of lines for each block
		int rem = *nline % size;
//		printf("here! div:%d\trem:%d", div, rem);

		int start = rem > 0 ? div + 1 : div;
		int startInit = start;	// save for cooking rank 0's own data
		// start from the machine with rank 1 (rank 0 machine keep a part of data)
		for (i = 1; i < size; i++) {
			// transfer data nearly equally
			int transSize = i < rem ? div + 1 : div;

			printf("rank:%d sending...size:%d\n", rank, transSize);
			// dst, tag, comm
			MPI_Send(data[start], ndim * transSize, MPI_FLOAT, i, i, comm);
			start += transSize;
		}

		// cook rank 0's own data
//		dataShard = data;
//		dataShard[0] = (float *) realloc(dataShard[0], startInit * ndim *sizeof(float));
//		dataShard = (float **) realloc(dataShard, startInit * sizeof(float *));	// pointers for each row
		*nline = startInit;
		dataShard = (float**) malloc(*nline * sizeof(float *));
		dataShard[0] = (float *) malloc(*nline * ndim * sizeof(float));
		for (i = 1; i < *nline; i++)
			dataShard[i] = dataShard[i-1] + ndim;

		memcpy(dataShard[0], data[0], *nline * ndim * sizeof(float));
	}
	// other processes
	else {
		int div = *nline / size;	// # of lines for each block
		int rem = *nline % size;
		*nline = rank < rem ? div + 1 : div;

		dataShard = (float**) malloc(*nline * sizeof(float *));
		dataShard[0] = (float *) malloc(*nline * ndim * sizeof(float));
		for (i = 1; i < *nline; i++) {
			dataShard[i] = dataShard[i-1] + ndim;
		}

		printf("rank:%d receiving..nline:%d\n", rank, *nline);
		// src, tag, comm, status
		MPI_Recv(dataShard[0], *nline * ndim, MPI_FLOAT, root, rank, comm, &status);
		printf("rank:%d received done\n", rank);
	}

	return dataShard;
}

int kmeans_write(char *outputfilename,
		int numberofLocalData, int numberofTotalData, int numberofClusters,
		int numberofCoordinates, float **clusters, int *localMemebership,
		int ranktooutput, MPI_Comm comm) {

	FILE *fp;
	MPI_Status mpistatus;

	int i, j;


	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	if (rank == ranktooutput) {
		printf("Start writing results (cluster centroid and membership) of K=%d cluster centers to file \"%s\"\n",
				numberofClusters, outputfilename);

//		err = MPI_File_open(MPI_COMM_SELF, outputfilename,
//				MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpif);
//		if (err != MPI_SUCCESS) {
//			printf("Error: Cannot access file \"%s\"\n",
//					outputfilename);
//			MPI_Finalize();
//			exit(1);
		fp = fopen(outputfilename, "w");
		if (fp == NULL ){
			printf("Error: cannot access the outputfile: %s\n", outputfilename);
			exit(EXIT_FAILURE);
		} else {
			for (i = 0; i < numberofClusters; i++) {
				char str[32];
				fprintf(fp, "%d ", i);
//				MPI_File_write(mpif, str, strlen(str), MPI_CHAR, &mpistatus);
				for (j = 0; j < numberofCoordinates; j++) {
					fprintf(fp, "%f ", clusters[i][j]);

				}
//				MPI_File_write(mpif, "\n", 1, MPI_CHAR, &mpistatus);
				fprintf(fp, "\n");
			}
		}
		printf("Finish writing clusters");
//		MPI_File_close(&mpif);
//
//		printf(
//				"Start writing clusters to which all %d data belonging to file \"%s\"\n",
//				numberofTotalData, filename_belongtocluster);
//		err = MPI_File_open(comm, filename_belongtocluster,
//				MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpif);
//		if (err != MPI_SUCCESS) {
//			printf("Error: Cannot access file \"%s\"\n",
//					filename_belongtocluster);
//			MPI_Finalize();
//			exit(1);
//		}

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
					fprintf(fp, "%d %d\n", outputCount++, localMemebership[j]);
//					MPI_File_write(mpif, str, strlen(str), MPI_CHAR,
//							&mpistatus);
				}
				continue;
			}
			MPI_Recv(localMemebership, numberofRecData, MPI_INT, i, i, comm,
					&mpistatus);
			//print out received membership
			for (j = 0; j < numberofRecData; j++) {
				fprintf(fp, "%d %d\n", outputCount++, localMemebership[j]);
//				MPI_File_write(mpif, str, strlen(str), MPI_CHAR, &mpistatus);
			}
		}
		fclose(fp);
	} else {
		MPI_Send(localMemebership, numberofLocalData, MPI_INT, ranktooutput,
				rank, comm);
	}
	return 1;
}

//for computing the Euclidean distance (for 2D data point)
float Compute_ED(float *datapoint1, float *datapoint2, int numberofCoordinates){
	float distance = 0;
	int i;
	for(i = 0; i < numberofCoordinates; i++){
		distance += (datapoint1[i] - datapoint2[i]) * (datapoint1[i] - datapoint2[i]);
	}
	return distance;
}

float Compute_DNADist(float *datapoint1, float *datapoint2, int numberofCoordinates){
	float distance = 0;
	int i;
	for(i = 0; i < numberofCoordinates; i++){
		distance += (datapoint1[i] == datapoint2[i]) ? 0 : 1;
	}
	return distance;
}

//for find the nearest neighbor in the given set;
int find_NN(int type, float *datapoint, float ** neighborset, int numberofNeighber,
		int numberofCoordinates) {
	int i;
	int nearest_neighbor = -1;
	float distance, mindist;
	mindist = FLT_MAX;
	for(i = 0; i < numberofNeighber; i++){
		if(type == NORMDATA)
			distance = Compute_ED(datapoint, neighborset[i], numberofCoordinates);
		else if(type == DNADATA)
			distance = Compute_DNADist(datapoint, neighborset[i], numberofCoordinates);
		if(distance < mindist){
			mindist = distance;
			nearest_neighbor = i;
		}
	}
	return nearest_neighbor;
}

int kmeans(int type, float **data, int numberofClusters, int numberofCoordinates,
		int numberofData, float stopthreshold, int *membership,
		float **clusters, MPI_Comm comm) {
	float **updatedClusters;
	int *** ClusterDNAcounts, ***tmpClusterDNAcounts;
	int *updatedClusterSize, *tmpClusterSize;

	int i, j, k, totaldifferences;;
	int rank;
	MPI_Comm_rank(comm, &rank);


	//get the total data number
	int numberofTotalData = 0;
	MPI_Allreduce(&numberofData, &numberofTotalData, 1, MPI_INT, MPI_SUM, comm);


	//initialization
	//malloc space for pointers
	updatedClusterSize = (int *) calloc(numberofClusters, sizeof(int));
	tmpClusterSize = (int *) calloc(numberofClusters, sizeof(int));

	updatedClusters = (float **) malloc(numberofClusters * sizeof(float*));
	updatedClusters[0] = (float *) calloc(numberofClusters * numberofCoordinates,
			sizeof(float));
	for (i = 1; i < numberofClusters; i++) {
		updatedClusters[i] = updatedClusters[i - 1] + numberofCoordinates;
	}

	//initiate the DNA counters
	ClusterDNAcounts = (int ***) malloc(numberofClusters * sizeof(int**));
	for(i = 0; i < numberofClusters; i++){
		ClusterDNAcounts[i] = (int **) malloc(numberofCoordinates *
				sizeof(int *));
		ClusterDNAcounts[i][0] = (int *) calloc(DNATYPENUM * numberofCoordinates,
						sizeof(int));
		for(j = 1; j < numberofCoordinates; j ++)
			ClusterDNAcounts[i][j] = ClusterDNAcounts[i][j - 1] + DNATYPENUM;
	}

	tmpClusterDNAcounts = (int ***) malloc(numberofClusters * sizeof(int**));
		for(i = 0; i < numberofClusters; i++){
			tmpClusterDNAcounts[i] = (int **) malloc(numberofCoordinates *
					sizeof(int *));
			tmpClusterDNAcounts[i][0] = (int *) calloc(DNATYPENUM * numberofCoordinates,
							sizeof(int));
			for(j = 1; j < numberofCoordinates; j ++)
				tmpClusterDNAcounts[i][j] = tmpClusterDNAcounts[i][j - 1] + DNATYPENUM;
	}

	if (!updatedClusterSize || !tmpClusterSize || !updatedClusters
			|| !updatedClusters[0]) {
		printf("Proc %d Error: Cannot calloc space for the new cluster variables", rank);
		exit(1);
	}

	//reset memeber ship
	for (i = 0; i < numberofData; i++){
		membership[i] = -1;
	}

	float delta;
	delta = FLT_MAX;
	int index, differences;
	int iterations = 0;
	while(delta > stopthreshold && iterations < MY_MAXITER){
		differences = 0;
		iterations ++;
		printf("Proc %d: iteration %d \n", rank, iterations);

		delta = 0.0;
		for(i = 0; i < numberofData; i++) {
			index = find_NN(type, data[i], clusters, numberofClusters, numberofCoordinates);
			if(index < 0){
				printf("Proc %d Error: mistake in finding nearest cluster.", rank);
				exit(1);
			}
			if(index != membership[i])differences ++;
			membership[i] = index;
			updatedClusterSize[index]++;

			//update the sum of the coordinates for 2D data points
			//which can be used for later usage
			if(type == NORMDATA){
				for(j = 0; j < numberofCoordinates; j++) {
					updatedClusters[index][j] += data[i][j];
				}
			}
			else if(type == DNADATA){
				for(j = 0; j < numberofCoordinates; j++){
//					printf("Proc %d: test-> %d\n", rank, (int)data[i][j] - 1);
					ClusterDNAcounts[index][j][(int)data[i][j] - 1]++;
				}
			}
		}

		//compute the new cluster centroid
		if(type == NORMDATA){
			//reduce all cluster's partial sums to the total sum for every cluster
			MPI_Allreduce(updatedClusters[0], clusters[0], numberofClusters * numberofCoordinates, MPI_FLOAT, MPI_SUM, comm);

			//reduce all cluster'size to get the total size for every cluster
			MPI_Allreduce(updatedClusterSize, tmpClusterSize, numberofClusters, MPI_INT, MPI_SUM, comm);
			for(i = 0; i < numberofClusters; i++){
				for(j = 0; j < numberofCoordinates; j++){
					if(tmpClusterSize[i] > 0){
						clusters[i][j] /= tmpClusterSize[i];
					}
					updatedClusters[i][j] = 0;
				}
				updatedClusterSize[i] = 0;
			}
		}
		else if(type == DNADATA){
			for(i = 0; i < numberofClusters; i++){
				MPI_Allreduce(ClusterDNAcounts[i][0], tmpClusterDNAcounts[i][0], numberofCoordinates * DNATYPENUM, MPI_INT, MPI_SUM, comm);
//				printf("Proc %d: received totoal dna count of cluster %d: ", rank, i);
//				for(k = 0; k < numberofCoordinates; k++){
//					printf("1:%d 2:%d 3:%d 4:%d, ", tmpClusterDNAcounts[i][k][0], tmpClusterDNAcounts[i][k][1], tmpClusterDNAcounts[i][k][2], tmpClusterDNAcounts[i][k][3]);
//				}
//				printf("\n");

				for(j = 0; j < numberofCoordinates; j++){
					int mostappearDNA = -1;
					int maxcount = 0;
					for(k = 0; k < DNATYPENUM; k ++){
						if(tmpClusterDNAcounts[i][j][k] > maxcount){
							mostappearDNA = k;
							maxcount = tmpClusterDNAcounts[i][j][k];
						}
						ClusterDNAcounts[i][j][k] = 0;
					}
					clusters[i][j] = mostappearDNA;
				}
//				printf("Proc %d: dna centrod: ", rank);
//				for(k = 0; k < numberofCoordinates; k++){
//					printf("%d ", (int)clusters[i][k]);
//				}
//				printf("\n");
			}

		}

		MPI_Allreduce(&differences, &totaldifferences, 1, MPI_INT, MPI_SUM, comm);
		delta = totaldifferences / (double) numberofTotalData;
	}

	free(updatedClusters[0]);
	free(updatedClusters);
	free(updatedClusterSize);
	free(tmpClusterSize);
	for(i = 0; i < numberofClusters; i++){
		free(ClusterDNAcounts[i][0]);
	}
	free(ClusterDNAcounts);
	for(i = 0; i < numberofClusters; i++){
			free(tmpClusterDNAcounts[i][0]);
		}
	free(tmpClusterDNAcounts);

	return 1;
}
