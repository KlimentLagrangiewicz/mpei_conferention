#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "help.h"
#include "kmeans.h"

int main(int argc, char **argv) {
	if (argc < 8) {
		puts("Not enough parameters...");
		exit(1);
	}
	const int n = atoi(argv[1]), m = atoi(argv[2]), k = atoi(argv[3]), l = atoi(argv[4]) ;
	if ((n < 0) || (m < 0) || (k < 0) || (k > n) || (l < 0) || (l > (n - 1))) {
		puts("Value of parameters is incorrect...");
		exit(1);
	}
	int i;
	int *sn = (int*)malloc(k * sizeof(int));
	for (i = 0; i < k; i++) {
		sn[i] = atoi(argv[8 + i]);
	}
	for (i = 0; i < k; i++) {
		printf("%d%c", sn[i], ((i + 1) == k) ? '\n' : ' ');
	}
	int *y1 = (int*)malloc(n * sizeof(int));
	int *y2 = (int*)malloc(n * sizeof(int));
	int *y3 = (int*)malloc(n * sizeof(int));
	double *x = (double*)malloc(n * m * sizeof(double));
	fscanfData(argv[5], x, n * m);
	kmeans(x, y1, sn, n, m, k);
	kmeansNN(x, y2, sn, n, m, k, l);
	kmeansEN(x, y3, sn, n, m, k, l);
	int *ideal = (int*)malloc(n * sizeof(int));
	fscanfSplitting(argv[6], ideal, n);
	const double a1 = getAccuracy(ideal, y1, n), a2 = getAccuracy(ideal, y2, n), a3 = getAccuracy(ideal, y3, n);
	double q1 = getQuality(x, y1, n, m), q2 = getQuality(x, y2, n, m), q3 = getQuality(x, y3, n, m);
	fprintfRes(argv[7], sn, n, m, k, l, a1, a2, a3, q1, q2, q3);
	printf("Accuracy of splitting:\nk-means = %.5lf\nk-means + NN = %.5lf\nk-means + EpsN = %.5lf\n"
		   "Quality of splitting:\nk-means = %.5lf\nk-means + NN = %.5lf\nk-means + EpsN = %.5lf\nThe program is ending...\n", a1, a2, a3, q1, q2, q3);
	free(x);
	free(y1);
	free(y2);
	free(y3);
	free(sn);
	free(ideal);
	return 0;
}
