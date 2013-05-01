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

int main(int argc, char **argv) {
	extern char *optarg;
	extern int optind, optopt;
	int c, ncluster = 4, rank, nworker, nline, totalLine, ndim, i = 0;
	char *inFile, *outFile;
	float thres = 0.01;

	float **data;	// input data in this process
	float **centroid;	// all cluster centroids
	int *label;	// for each data point, find its new class label

	double stime, etime;

	while ((c = getopt(argc, argv, "i:o:k:t:l:d:")) != EOF) {
		switch (c) {
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
			fprintf(stderr, "Usage: %s -f inFile -k #OfClusters -t threshold -l #ofLinesInInputFile -d #dimension\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	if (optind >= argc) {
		fprintf(stderr, "Usage: %s -f inFile -k #OfClusters -t threshold -l #ofLinesInInputFile -d #dimension\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	stime = MPI_Wtime();

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nworker);

	// read input data (each process has a portion of all data)
	// IMP: nline changed from total lines for all input to subtotal lines for a specific process
	totalLine = nline;
	data = kmeans_read(inFile, &nline, ndim, MPI_COMM_WORLD);

	// initialize cluster centers
//	centroid = (float **) malloc(ncluster * sizeof(float));	// pointer to each line
//	centroid[0] = (float *) malloc(ncluster * ndim * sizeof(float));
//	for(i = 0; i < ncluster; i++)
//		centroid[i] = centroid[i-1] + ndim;
	centroid[ncluster][ndim];
	if(rank == 0)
		memcpy(centroid, data, ncluster * ndim * sizeof(float));

	// broadcast the centroid to all other processes
	MPI_Bcast(centroid[0], ncluster*ndim, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// do kmeans calculation
	label[nline];
	kmeans(data, ndim, nline, ncluster, thres, label, centroid, MPI_COMM_WORLD);

	// write cluster centroids to disk
	// TODO 2 write to same outFile ?
	kmean_write(outFile, outFile, nline, totalLine, ncluster,	ndim , centroid, label, 0, MPI_COMM_WORLD);

	etime = MPI_Wtime();

	// performance report
	printf("Done! Rank: %d\tTime: %f\n", rank, etime-stime);

	MPI_Finalize();
	return EXIT_SUCCESS;
}

