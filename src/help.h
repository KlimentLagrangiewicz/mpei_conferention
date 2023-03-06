#ifndef HELP_H_
#define HELP_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void fscanfData(const char *fn, double *x, const int n);
void fprintfRes(const char *fn, const int *sn, const int n, const int m, const int k, const int l, const double a1, const double a2, const double a3,
				const double q1, const double q2, const double q3);
void fscanfSplitting(const char *fn, int *y, const int n);
double getAccuracy(const int *ideal, const int *r, const int n);
double getQuality(const double *X, const int *y, const int n, const int m);

#endif
