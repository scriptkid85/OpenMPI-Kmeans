/*
 * openMPI-Kmeans.h
 *
 *  Created on: Apr 29, 2013
 *      Author: guanyuw
 */

#ifndef OPENMPI_KMEANS_H_
#define OPENMPI_KMEANS_H_

float** kmeans_read();
int     kmeans_write(char*, char*, int, int, int, int, float**, int*, int, MPI_Comm);
int     kmeans(float**, int, int, int, float, int*, float**, MPI_Comm);


#endif /* OPENMPI_KMEANS_H_ */
