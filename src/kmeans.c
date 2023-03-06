#include "kmeans.h"

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

static int getCluster(const double *x, const double *c, const int m, const int k) {
	double curD, minD = DBL_MAX;
	int counter, res;
	counter = res = 0;
	while (counter < k) {
		curD = getEvDist(x, c, m);
		if (curD < minD) {
			minD = curD;
			res = counter;
		}
		counter++;
		c += m;
	}
	return res;
}

static void detCores(const double *x, double *c, const int *sn, const int k, const int  m) {
	int i, j, buf1, buf2;
	for (i = 0; i < k; i++) {
		buf1 = i * m;
		buf2 = sn[i] * m;
		for (j = 0; j < m; j++) {
			c[buf1 + j] = x[buf2 + j];
		}
	}
}

static void detStartSplitting(const double *x, const double *c, int *y, int *nums, const int n, const int m, const int k) {
	int i = 0, j = 0, cur;
	while (i < n) {
		cur = getCluster(&x[j], &c[0], m, k);
		y[i] = cur;
		nums[cur]++;
		j += m;
		i++;
	}
}

static void calcCores(const double *x, double *c, const int *res, const int *nums, const int n, const int m) {
	int i, j, buf1, buf2, buf3;
	for (i = 0; i < n; i++) {
		buf1 = nums[res[i]];
		buf2 = res[i] * m;
		buf3 = i * m;
		for (j = 0; j < m; j++) {
			c[buf2 + j] += x[buf3 + j] / buf1;
		}
	}
}

static char checkSplitting(const double *x, const double *c, int *res, int *nums, const int n, const int m, const int k) {
	int i = 0, count = 0, j = 0, f;
	while (i < n) {
		f = getCluster(&x[j], &c[0], m, k);
		if (f == res[i]) count++;
		res[i] = f;
		nums[f]++;
		j += m;
		i++;
	}
	return (n == count) ? 0 : 1;
}


void kmeans(const double *X, int *y, const int *sn, const int n, const int m, const int k) {
	int *nums = (int*)malloc(k * sizeof(int));
	memset(nums, 0, k * sizeof(int));
	double *x = (double*)malloc(n * m * sizeof(double));
	double *c = (double*)malloc(k * m * sizeof(double));
	memcpy(x, X, n * m * sizeof(double));
	autoscaling(x, n, m);
	detCores(x, c, sn, k, m);
	detStartSplitting(x, c, y, nums, n, m, k);
	char flag = 1;
	int i = 0;
	do {
		memset(c, 0, k * m * sizeof(double));
		calcCores(x, c, y, nums, n, m);
		memset(nums, 0, k * sizeof(int));
		flag = checkSplitting(x, c, y, nums, n, m, k);
		i++;
	} while ((flag) || (i < 2));
	free(x);
	free(c);
	free(nums);
}

static void getNeighborsMatrix(const double *x, int *nums, const int n, const int m, const int l) {
	char *v = (char*)malloc(n * sizeof(char));
	int i, j, k, id, buf;
	double curD, minD;
	for (i = 0; i < n; i++) {
		buf = i * m;
		memset(v, 0, n * sizeof(char));
		v[i] = 1;
		for (k = 0; k < l; k++) {
			j = 0;
			while (v[j] && (j < n)) j++;
			id = j;
			minD = DBL_MAX;
			while (j < n) {
				curD = getEvDist(&x[buf], &x[j * m], m);
				if ((curD < minD) && (v[j] == 0)) {
					minD = curD;
					id = j;
				}
				j++;
			}
			v[id] = 1;
			nums[i * l + k] = id;
		}
	}
	free(v);
}

static char checkSplittingNN(int *y, const int *nums, const int n, const int k, const int l) {
	int *fr = (int*)malloc(k * sizeof(int));
	int counter = 0, i, j, maxFr, id;
	for (i = 0; i < n; i++) {
		memset(fr, 0, k * sizeof(int));
		id = y[i];
		fr[id] = 1;
		for (j = i * l; j < i * l + l; j++) {
			fr[y[nums[j]]]++;
		}
		maxFr = fr[id];
		for (j = 0; j < k; j++) {
			if (fr[j] > maxFr) {
				maxFr = fr[j];
				id = j;
			}
		}
		if (id == y[i]) counter++;
		y[i] = id;
	}
	free(fr);
	return (counter == n) ? 0 : 1;
}

/*
static void printfMatr(const int *x, const int n, const int m) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = i * m; j < i * m + m; j++) {
			printf("%d%c", x[j], (((j + 1) % m) == 0) ? '\n' : ' ');
		}
	}
}
*/

void kmeansNN(const double *X, int *y, const int *sn, const int n, const int m, const int k, const int l) {
	int *nums = (int*)malloc(k * sizeof(int));
	memset(nums, 0, k * sizeof(int));
	double *x = (double*)malloc(n * m * sizeof(double));
	double *c = (double*)malloc(k * m * sizeof(double));
	memcpy(x, X, n * m * sizeof(double));
	autoscaling(x, n, m);
	detCores(x, c, sn, k, m);
	detStartSplitting(x, c, y, nums, n, m, k);
	char flag = 1;
	do {
		memset(c, 0, k * m * sizeof(double));
		calcCores(x, c, y, nums, n, m);
		memset(nums, 0, k * sizeof(int));
		flag = checkSplitting(x, c, y, nums, n, m, k);
	} while (flag);
	free(c);
	/* correction is starting here */
	nums = (int*)realloc(nums, n * l * sizeof(int));
	getNeighborsMatrix(x, nums, n, m, l);
	flag = 1;
	do {
		flag = checkSplittingNN(y, nums, n, k, l);
	} while (flag);
	free(nums);
	free(x);
}

static void siftDown(double *numbers, int root, const int bottom) {
	double temp;
	int maxChild;
	short done = 0;
	while ((root * 2 <= bottom) && (!done)) {
		if (root * 2 == bottom)
			maxChild = root * 2;
		else
			if (numbers[root * 2] > numbers[root * 2 + 1])
				maxChild = root * 2;
			else
				maxChild = root * 2 + 1;
		if (numbers[root] < numbers[maxChild]) {
			temp = numbers[root];
			numbers[root] = numbers[maxChild];
			numbers[maxChild] = temp;
			root = maxChild;
		} else done = 1;
	}
}

static void heapSort(double *a, const int n) {
	int i;
	double temp;
	for (i = (n / 2); i >= 0; i--)
		siftDown(a, i, n - 1);
	for (i = n - 1; i >= 1; i--) {
		temp = a[0];
		a[0] = a[i];
		a[i] = temp;
		siftDown(a, 0, i - 1);
	}
}

static double getEpsVal(const double *x, const int *nums, const int n, const int m, const int l) {
	double *dist = (double*)malloc(n * sizeof(double));
	double d;
	int i, j;
	for (i = 0; i < n; i++) {
		d = 0;
		for (j = i * l; j < i * l + l; j++) {
			d += pow(getEvDist(&x[i * m], &x[nums[j] * m], m), l);
		}
		d /= l;
		d = pow(d, 1.0 / l);
		dist[i] = d;
	}
	heapSort(dist, n);
	d = ((n % 2) == 0) ? (0.5 * (dist[n / 2 - 1] + dist[n / 2])) : dist[n / 2];
	free(dist);
	return d;
}

static void getRelMatr(const double *x, char *nm, const int n, const int m, const double eps) {
	int i, j, k;
	i = k = 0;
	while (i < n * m) {
		j = 0;
		while (j < n * m) {
			nm[k] = (getEvDist(&x[i], &x[j], m) > eps) ? 0 : 1;
			k++;
			j += m;
		}
		i += m;
	}
}

static char checkSplittingEN(int *y, const char *nm, const int n, const int k) {
	int *fr = (int*)malloc(k * sizeof(int));
	int counter = 0, i, j, maxFr, id;
	for (i = 0; i < n; i++) {
		memset(fr, 0, k * sizeof(int));
		id = i * n;
		for (j = 0; j < n; j++) {
			if (nm[id + j]) fr[y[j]]++;
		}
		id = y[i];
		maxFr = fr[id];
		for (j = 0; j < k; j++) {
			if (fr[j] > maxFr) {
				maxFr = fr[j];
				id = j;
			}
		}
		if (id == y[i]) counter++;
		y[i] = id;
	}
	free(fr);
	return (counter == n) ? 0 : 1;
}

void kmeansEN(const double *X, int *y, const int *sn, const int n, const int m, const int k, const int l) {
	int *nums = (int*)malloc(k * sizeof(int));
	memset(nums, 0, k * sizeof(int));
	double *x = (double*)malloc(n * m * sizeof(double));
	double *c = (double*)malloc(k * m * sizeof(double));
	memcpy(x, X, n * m * sizeof(double));
	autoscaling(x, n, m);
	detCores(x, c, sn, k, m);
	detStartSplitting(x, c, y, nums, n, m, k);
	char flag = 1;
	do {
		memset(c, 0, k * m * sizeof(double));
		calcCores(x, c, y, nums, n, m);
		memset(nums, 0, k * sizeof(int));
		flag = checkSplitting(x, c, y, nums, n, m, k);
	} while (flag);
	free(c);
	/* correction is starting here */
	nums = (int*)realloc(nums, n * l * sizeof(int));
	getNeighborsMatrix(x, nums, n, m, l);
	const double eps = getEpsVal(x, nums, n, m, l);
	char *nm = (char*)malloc(n * n * sizeof(char));
	getRelMatr(x, nm, n, m, eps);
	flag = 1;
	do {
		flag = checkSplittingEN(y, nm, n, k);
	} while (flag);
	free(nm);
	free(nums);
	free(x);
}
