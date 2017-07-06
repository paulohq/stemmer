#include <stdio.h>
//#include "test.h"

__global__ void stemmer(char *d_out, char *d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = (f * f) +100;
}
