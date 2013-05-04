/*
 ============================================================================
 Name        : openMPI-main.c
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
#include <unistd.h>
#include "openMPI-Kmeans.h"

void printUsage(char *name) {
	fprintf(stderr, "Usage: %s -i inFile -o outFile -k #OfClusters -t threshold -l #ofLinesInInputFile -d #dimension\n", name);
}

int main(int argc, char **argv) {
	extern char *optarg;
	extern int optind, optopt;
	int c, ncluster = 4, rank, nworker, nline, totalLine, ndim, i = 0, j=0;
	int type;
	char *inFile, *outFile;
	float thres = 0.01;

	float **data;	// input data in this process
	float **centroid;	// all cluster centroids
	int *label;	// for each data point, find its new class label

	double stime, etime, 	// whole system time
		stimeCluster, etimeCluster;	// maximum cluster time among all processes
	double elapse, elapseWhole, elapseCluster, elapseClusterWhole;

	if(argc != 15) {
		printUsage(argv[0]);
		exit(EXIT_FAILURE);
	}

	while ((c = getopt(argc, argv, "p:i:o:k:t:l:d:")) != EOF) {
		switch (c) {
		case 'p':
			type = atoi(optarg);
			break;
		case 'i':
			inFile = optarg;
			break;
		case 'o':
			outFile = optarg;
			break;
		case 'k':
			ncluster = atoi(optarg);
			break;
		case 't':
			thres = atof(optarg);
			break;
		case 'l':
			nline = atof(optarg);
			break;
		case 'd':
			ndim = atof(optarg);
			break;
		default:
			printUsage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	if (optind > argc) {
		printUsage(argv[0]);
		exit(EXIT_FAILURE);
	}

	stime = MPI_Wtime();

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nworker);
	printf("rank:%d init done\n", rank);

	// read input data (each process has a portion of all data)
	// IMP: nline changed from total lines for all input to subtotal lines for a specific process
	totalLine = nline;
	data = kmeans_read(inFile, &nline, ndim, MPI_COMM_WORLD);
	printf("rank:%d read data done. nline:%d\n", rank, nline);

	// initialize cluster centers
	centroid = (float **) malloc(ncluster * sizeof(float *));	// pointer to each line
	centroid[0] = (float *) malloc(ncluster * ndim * sizeof(float));
	for(i = 1; i < ncluster; i++)
		centroid[i] = centroid[i-1] + ndim;
	//float centroid[ncluster][ndim];
	if(rank == 0) {
		for(i = 0; i < ncluster; i++)
			for(j = 0; j < ndim; j++)
				centroid[i][j] = data[i][j];
	}
//		memcpy(centroid, data, ncluster * ndim * sizeof(float));
	printf("rank:%d init cluster center done\n", rank);

	// broadcast the centroid to all other processes
	MPI_Bcast(centroid[0], ncluster*ndim, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// do kmeans calculation
	stimeCluster = MPI_Wtime();
	label = (int *) malloc(nline * sizeof(int));
	//printf("rank:%d prior kmeans\n", rank);
	for(i = 0; i < ncluster; i++){
			printf("Main Proc %d cluster %d: ", rank, i);
			for(j = 0; j < ndim; j++){
				printf("%f ", centroid[i][j]);
			}
			printf("\n");
		}
	kmeans(type, data, ncluster, ndim, nline, thres, label, centroid, MPI_COMM_WORLD);
	printf("rank:%d kmeans done\n", rank);
	etimeCluster = MPI_Wtime();

	// write cluster centroids to disk
	// TODO 2 write to same outFile ?
	kmeans_write(outFile, nline, totalLine, ncluster, ndim , centroid, label, 0, MPI_COMM_WORLD);

	free(label);
	free(centroid[0]);
	free(centroid);
	etime = MPI_Wtime();
	elapse = etime - stime;
	elapseCluster = etimeCluster-stimeCluster;

	// get the maximum time among processes
	MPI_Reduce(&elapse, &elapseWhole, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&elapseCluster, &elapseClusterWhole, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	// performance report
	printf("Done! Rank: %d\tTime: %f\n", rank, etime-stime);
	if(rank == 0)
		printf("System time: %f secs\tClustering time: %fsecs\n", elapseWhole, elapseCluster);

	MPI_Finalize();
	return EXIT_SUCCESS;
}
