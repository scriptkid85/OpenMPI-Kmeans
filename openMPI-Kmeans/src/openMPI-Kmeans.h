/*
 * openMPI-Kmeans.h
 *
 *  Created on: Apr 29, 2013
 *      Author: guanyuw
 */

#ifndef OPENMPI_KMEANS_H_
#define NORMDATA 0
#define DNADATA 1
#define OPENMPI_KMEANS_H_



float** kmeans_read(char*, int*, int, MPI_Comm);
int     kmeans_write(char*, int, int, int, int, float**, int*, int, MPI_Comm);
int     kmeans(int, float**, int, int, int, float, int*, float**, MPI_Comm);


#endif /* OPENMPI_KMEANS_H_ */
