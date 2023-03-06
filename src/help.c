#include "help.h"

void fscanfData(const char *fn, double *x, const int n) {
	FILE *fl = fopen(fn, "r");
	if (fl == NULL) {
		printf("Error in opening %s file..\n", fn);
		exit(1);
	}
	int i = 0;
	while ((i < n) && (!feof(fl))) {
		if (fscanf(fl, "%lf", &x[i]) == 0) {}
		i++;
	}
	fclose(fl);
}

void fprintfRes(const char *fn, const int *sn, const int n, const int m, const int k, const int l, const double a1, const double a2, const double a3,
				const double q1, const double q2, const double q3) {
	FILE *fl  = fopen(fn, "a");
	if (fl == NULL) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	fprintf(fl, "Result of testing algorithms...\nN = %d;\nM = %d;\nK = %d;\nL = %d;\nStart numbers = {", n, m, k, l);
	int i;
	for (i = 0; i < k; i++) {
		fprintf(fl, "%d%c%c", sn[i], ((i + 1) == k) ? '}' : ',', ((i + 1) == k) ? '\n' : ' ');

	}
	fprintf(fl,	"Accuracy of splitting:\nk-means = %.5lf\nk-means + NN = %.5lf\nk-means + EpsN = %.5lf\n"
				"Quality of splitting:\nk-means = %.5lf\nk-means + NN = %.5lf\nk-means + EpsN = %.5lf\n\n",
				a1, a2, a3, q1, q2, q3);
	fclose(fl);
}

void fscanfSplitting(const char *fn, int *y, const int n) {
	FILE *fl = fopen(fn, "r");
	if (fl == NULL) {
		printf("Can't access %s file with ideal splitting for reading...\n", fn);
		exit(1);
	}
	int i = 0;
	while ((i < n) && !feof(fl)) {
		if (fscanf(fl, "%d", &y[i]) == 0) {}
		i++;
	}
	fclose(fl);
}

static int getNumOfClass(const int *y, const int n) {
	int i, j, cur;
	char *v = (char*)malloc(n * sizeof(char));
	memset(v, 0, n * sizeof(char));
	for (i = 0; i < n; i++) {
		while ((v[i]) && (i < n)) i++;
		cur = y[i];
		for (j = i + 1; j < n; j++) {
			if (y[j] == cur)
				v[j] = 1;
		}
	}
	i = cur = 0;
	while (i < n) {
		if (v[i] == 0) cur++;
		i++;
	}
	free(v);
	return cur;
}

static double getCurAccuracy(const int *x, const int *y, const int *a, const int n) {
	int i, j;
	i = j = 0;
	while  (i < n) {
		if (x[i] == a[y[i]]) j++;
		i++;
	}
	return (double)j / (double)n;
}

static void solve(const int *x, const int *y, int *items, int size, int l, const int n, double *eps) {
    int i;
    if (l == size) {
    	double cur = getCurAccuracy(x, y, items, n);
    	if (cur > *eps) *eps = cur;
    } else {
        for (i = l; i < size; i++) {
            if (l ^ i) {
            	items[l] ^= items[i];
            	items[i] ^= items[l];
            	items[l] ^= items[i];
            	solve(x, y, items, size, l + 1, n, eps);
            	items[l] ^= items[i];
            	items[i] ^= items[l];
            	items[l] ^= items[i];
            } else {
            	solve(x, y, items, size, l + 1, n, eps);
            }
        }
    }
}

double getAccuracy(const int *ideal, const int *r, const int n) {
	const int k = getNumOfClass(ideal, n);
	int *nums = (int*)malloc(k * sizeof(int));
	int i = 0;
	while (i < k) {
		nums[i] = i;
		i++;
	}
	double max = getCurAccuracy(r, ideal, nums, n);
	i = 0;
	solve(r, ideal, nums, k, i, n, &max);
	free(nums);
	return max;
}

static double getEvDist(const double *x1, const double *x2, const int m) {
	double d, r = 0;
	int i = 0;
	while (i++ < m) {
		d = *(x1++) - *(x2++);
		r += d * d;
	}
	return sqrt(r);
}

static void autoscaling(double *x, const int n, const int m) {
	const int s = n * m;
	double sd, Ex, Exx;
	int i, j = 0;
	while (j < m) {
		i = j;
		Ex = Exx = 0;
		while (i < s) {
			sd = x[i];
			Ex += sd;
			Exx += sd * sd;
			i += m;
		}
		Exx /= n;
		Ex /= n;
		sd = sqrt(Exx - Ex * Ex);
		i = j;
		while (i < s) {
			x[i] = (x[i] - Ex) / sd;
			i += m;
		}
		j++;
	}
}

double getQuality(const double *X, const int *y, const int n, const int m) {
	double *x = (double*)malloc(n * m * sizeof(double));
	memcpy(x, X, n * m * sizeof(double));
	autoscaling(x, n, m);
	int i, j, buf, k1 = 0, k2 = 0;
	double inside = 0, outside = 0;
	for (i = 0; i < n; i++) {
		buf = i * m;
		for (j = i + 1; j < n; j++)  {
			if (y[i] == y[j]) {
				inside += getEvDist(&x[buf], &x[j * m], m);
				k1++;
			} else {
				outside += getEvDist(&x[buf], &x[j * m], m);
				k2++;
			}
		}
	}
	inside /= (k1 == 0) ? 1 : k1;
	outside /= (k2 == 0) ? 1 : k2;
	free(x);
	return (inside / ((outside == 0) ? inside : outside));
}
