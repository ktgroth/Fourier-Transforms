
#ifndef FFT_H
#define FFT_H

#include <complex.h>

#define M_PI 3.14159265358979323846

#include <tgmath.h>

#define MAX 1024 * 1024

#define ifft(f, N) (fft(f, N));
void fft(double complex *f, int N);

#endif
