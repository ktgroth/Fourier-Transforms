
#include <stdio.h>
#include <stdlib.h>

#include "include/fft.h"

void fft(double complex *f, int N)
{
    if (N <= 1)
        return;

    double complex *even = malloc(N / 2 * sizeof(double complex));
    double complex *odd = malloc(N / 2 * sizeof(double complex));

    for (int i = 0; i < N / 2; ++i)
    {
        even[i] = f[2 * i];
        odd[i] = f[2 * i + 1];
    }

    fft(even, N / 2);
    fft(odd, N / 2);

    for (int k = 0; k < N / 2; ++k)
    {
        double complex s = odd[k] * exp(-2.0 * _Complex_I * M_PI * k / N);
        f[k] = even[k] + s;
        f[k + N / 2] = even[k] - s;
    }

    free(even);
    free(odd);
}
