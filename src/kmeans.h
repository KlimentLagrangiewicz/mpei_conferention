#ifndef KMEANS_H_
#define KMEANS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

void kmeans(const double *X, int *y, const int *sn, const int n, const int m, const int k);
void kmeansNN(const double *X, int *y, const int *sn, const int n, const int m, const int k, const int l);
void kmeansEN(const double *X, int *y, const int *sn, const int n, const int m, const int k, const int l);

#endif
