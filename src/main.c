
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <complex.h>
#include <tgmath.h>

#include "include/fft.h"

typedef struct
{
    char     riff_chuck[4];
    uint32_t chunk_size;
    char     file_format[4];
    char     format_chunk[4];
    uint32_t format_size;
    uint16_t pcm_flags;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bit_depth;
    char     data_chunk[4];
    uint32_t data_size;
} wavfileheader_t;


int next_power_of_two(int n)
{
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

int main()
{
    double Fs = 1000;
    double T = 1 / Fs;
    uint32_t L = 1 << 10;

    double signal[L];
    double x = 0;
    double dx = 1. / L;

    for (uint64_t i = 0; i < L; ++i)
    {
        signal[i] = -sin(180 * x * 2 * M_PI) - cos(90 * x * 2 * M_PI);
        x += dx;
    }
    
    
    uint32_t fft_size = next_power_of_two(L);

    double complex f[fft_size];
    for (uint32_t i = 0; i < L; ++i)
    {
        __real__ f[i] = signal[i] * T;
        __imag__ f[i] = 0.0;
    }

    fft(f, fft_size);
    FILE *plt = fopen("fft_output.dat", "w");
    
    for (uint32_t i = 0; i < fft_size / 2; ++i)
    {
        double freq = i * Fs / fft_size;
        double magnitude = cabs(f[i]);
        fprintf(plt, "%f %f\n", freq, magnitude);
    }
    fprintf(plt, "e\n");
    fprintf(plt, "pause -1\n");
    fflush(plt);

    return 0;
}
