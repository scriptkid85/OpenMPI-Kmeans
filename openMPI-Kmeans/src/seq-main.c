/*
 * runKMeans.c
 *
 *  Created on: May 1, 2013
 *      Author: lzy
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>       /* time_t, struct tm, difftime, time, mktime */
#include "seq-Kmeans.h"


void printUsage(char *name) {
	fprintf(stderr, "Usage: %s -i inFile -o outFile -k #OfClusters -t threshold -l #ofLinesInInputFile -d #dimension\n", name);
}

int main(int argc, char **argv) {
	extern char *optarg;
	extern int optind, optopt;
	int c, ncluster = 4, nline, totalLine, ndim, i = 0;
	char *inFile, *outFile;
	float thres = 0.01;

	float **data;	// input data in this process
	float **centroid;	// all cluster centroids
	int *label;	// for each data point, find its new class label

	clock_t stime, etime, 	// whole system time
		stimeCluster, etimeCluster;	// maximum cluster time among all processes

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
			printUsage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	if (optind > argc) {
		printUsage(argv[0]);
		exit(EXIT_FAILURE);
	}

	stime = clock();

	printf("init done\n");

	// read input data (each process has a portion of all data)
	totalLine = nline;
	data = kmeans_read(inFile, nline, ndim);
	printf("read data done.\n");

	// initialize cluster centers
	centroid = (float **) malloc(ncluster * sizeof(float *));	// pointer to each line
	centroid[0] = (float *) malloc(ncluster * ndim * sizeof(float));
	for(i = 1; i < ncluster; i++)
		centroid[i] = centroid[i-1] + ndim;

	memcpy(centroid, data, ncluster * ndim * sizeof(float));
	printf("init cluster center done\n");

	// do kmeans calculation
	stimeCluster = clock();
	label = (int *) malloc(nline * sizeof(int));
	//printf("rank:%d prior kmeans\n", rank);
	kmeans(data, ndim, nline, ncluster, thres, label, centroid);
	printf("kmeans done\n");
	etimeCluster = clock();

	// write cluster centroids to disk
	// TODO 2 write to same outFile ?
	kmeans_write(outFile, outFile, nline, totalLine, ncluster, ndim , centroid, label, 0);

	free(label);
	free(centroid[0]);
	free(centroid);
	etime = clock();

	// performance report
	printf("System time: %f\tClustering time: %f\n", (double)(etime-stime) / (CLOCKS_PER_SEC * 1000), (double)(etimeCluster-stimeCluster) / (CLOCKS_PER_SEC * 1000));

	return EXIT_SUCCESS;
}
