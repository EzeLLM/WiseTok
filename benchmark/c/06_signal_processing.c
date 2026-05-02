/*
 * signal_processing.c - Digital Signal Processing implementation
 *
 * Includes FFT (radix-2), FIR/IIR filters, window functions,
 * and SIMD optimizations for x86 (SSE/AVX).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define M_PI 3.14159265358979323846
#define FFT_MAX_SIZE 1024

typedef struct {
  float real;
  float imag;
} complex_f32_t;

typedef struct {
  float *coeffs;
  float *state;
  int order;
} fir_filter_t;

typedef struct {
  float *b;    /* Forward coefficients */
  float *a;    /* Feedback coefficients */
  float *state;
  int order;
} iir_filter_t;

/**
 * window_hamming - Generate Hamming window function
 * @window: Output buffer for window values (must be size n)
 * @n: Window size
 *
 * Computes the Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k/(n-1))
 */
static void window_hamming(float *window, int n)
{
  for (int k = 0; k < n; k++) {
    float val = (float)k / (float)(n - 1);
    window[k] = 0.54f - 0.46f * cosf(2.0f * M_PI * val);
  }
}

/**
 * window_hann - Generate Hann (Hanning) window function
 * @window: Output buffer for window values
 * @n: Window size
 *
 * Computes the Hann window: w[k] = 0.5 - 0.5 * cos(2*pi*k/(n-1))
 */
static void window_hann(float *window, int n)
{
  for (int k = 0; k < n; k++) {
    float val = (float)k / (float)(n - 1);
    window[k] = 0.5f - 0.5f * cosf(2.0f * M_PI * val);
  }
}

/**
 * fft_radix2 - Cooley-Tukey radix-2 FFT
 * @x: Input/output buffer (complex numbers, in-place)
 * @n: FFT size (must be power of 2)
 *
 * Performs in-place Cooley-Tukey radix-2 FFT on complex input.
 */
static void fft_radix2(complex_f32_t *x, int n)
{
  if (n <= 1) return;

  /* Bit-reversal permutation */
  for (int i = 0; i < n; i++) {
    int j = 0;
    int k = i;
    for (int l = 1; l < n; l <<= 1) {
      j = (j << 1) | (k & 1);
      k >>= 1;
    }
    if (j > i) {
      complex_f32_t tmp = x[i];
      x[i] = x[j];
      x[j] = tmp;
    }
  }

  /* Butterfly passes */
  for (int s = 1; s <= (int)log2(n); s++) {
    int m = 1 << s;
    int m_half = m >> 1;

    for (int k = 0; k < n; k += m) {
      for (int j = 0; j < m_half; j++) {
        float angle = -2.0f * M_PI * j / m;
        float w_real = cosf(angle);
        float w_imag = sinf(angle);

        int idx1 = k + j;
        int idx2 = k + j + m_half;

        float t_real = w_real * x[idx2].real - w_imag * x[idx2].imag;
        float t_imag = w_real * x[idx2].imag + w_imag * x[idx2].real;

        x[idx2].real = x[idx1].real - t_real;
        x[idx2].imag = x[idx1].imag - t_imag;
        x[idx1].real += t_real;
        x[idx1].imag += t_imag;
      }
    }
  }
}

/**
 * fir_create - Create FIR filter
 * @coeffs: Filter coefficients
 * @order: Filter order
 *
 * Allocates and initializes an FIR filter structure.
 */
static fir_filter_t *fir_create(float *coeffs, int order)
{
  fir_filter_t *f = malloc(sizeof(fir_filter_t));
  if (!f) return NULL;

  f->coeffs = malloc(order * sizeof(float));
  f->state = calloc(order, sizeof(float));

  if (!f->coeffs || !f->state) {
    free(f->coeffs);
    free(f->state);
    free(f);
    return NULL;
  }

  memcpy(f->coeffs, coeffs, order * sizeof(float));
  f->order = order;

  return f;
}

/**
 * fir_process - Apply FIR filter to input signal
 * @f: FIR filter
 * @input: Input sample
 *
 * Processes a single sample through the FIR filter using direct form I.
 * Returns the filtered output.
 */
static float fir_process(fir_filter_t *f, float input)
{
  float output = 0.0f;

  /* Shift state and insert new sample */
  for (int i = f->order - 1; i > 0; i--) {
    f->state[i] = f->state[i - 1];
  }
  f->state[0] = input;

  /* Compute convolution using SSE if available */
  __m128 sum = _mm_set1_ps(0.0f);

  for (int i = 0; i < f->order - 3; i += 4) {
    __m128 coeff = _mm_loadu_ps(&f->coeffs[i]);
    __m128 state = _mm_loadu_ps(&f->state[i]);
    __m128 prod = _mm_mul_ps(coeff, state);
    sum = _mm_add_ps(sum, prod);
  }

  /* Horizontal sum of __m128 */
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  output = _mm_cvtss_f32(sum);

  /* Process remaining samples */
  int remainder = f->order % 4;
  for (int i = f->order - remainder; i < f->order; i++) {
    output += f->coeffs[i] * f->state[i];
  }

  return output;
}

/**
 * fir_free - Deallocate FIR filter
 * @f: Filter to free
 */
static void fir_free(fir_filter_t *f)
{
  if (f) {
    free(f->coeffs);
    free(f->state);
    free(f);
  }
}

/**
 * iir_create - Create IIR filter
 * @b: Forward coefficients (numerator)
 * @a: Feedback coefficients (denominator)
 * @order: Filter order
 *
 * Allocates and initializes an IIR filter structure.
 */
static iir_filter_t *iir_create(float *b, float *a, int order)
{
  iir_filter_t *f = malloc(sizeof(iir_filter_t));
  if (!f) return NULL;

  f->b = malloc(order * sizeof(float));
  f->a = malloc(order * sizeof(float));
  f->state = calloc(order, sizeof(float));

  if (!f->b || !f->a || !f->state) {
    free(f->b);
    free(f->a);
    free(f->state);
    free(f);
    return NULL;
  }

  memcpy(f->b, b, order * sizeof(float));
  memcpy(f->a, a, order * sizeof(float));
  f->order = order;

  return f;
}

/**
 * iir_process - Apply IIR filter to input signal
 * @f: IIR filter
 * @input: Input sample
 *
 * Processes a single sample through the IIR filter using direct form II.
 * Returns the filtered output.
 */
static float iir_process(iir_filter_t *f, float input)
{
  float y = 0.0f;
  float w = input;

  /* Compute feedback branch */
  for (int i = 1; i < f->order; i++) {
    w -= f->a[i] * f->state[i];
  }

  /* Compute forward branch */
  for (int i = 0; i < f->order; i++) {
    y += f->b[i] * (i == 0 ? w : f->state[i]);
  }

  /* Update state */
  for (int i = f->order - 1; i > 0; i--) {
    f->state[i] = f->state[i - 1];
  }
  f->state[0] = w;

  return y;
}

/**
 * iir_free - Deallocate IIR filter
 * @f: Filter to free
 */
static void iir_free(iir_filter_t *f)
{
  if (f) {
    free(f->b);
    free(f->a);
    free(f->state);
    free(f);
  }
}

/**
 * power_spectrum - Compute power spectrum from FFT output
 * @fft_data: FFT result (complex values)
 * @spectrum: Output power spectrum (magnitude squared)
 * @n: FFT size
 */
static void power_spectrum(complex_f32_t *fft_data, float *spectrum, int n)
{
  for (int i = 0; i < n; i++) {
    float real = fft_data[i].real;
    float imag = fft_data[i].imag;
    spectrum[i] = (real * real + imag * imag) / (n * n);
  }
}

/**
 * db_magnitude - Convert linear magnitude to dB scale
 * @lin: Linear magnitude
 *
 * Converts linear magnitude to decibels: 20 * log10(x)
 */
static float db_magnitude(float lin)
{
  return 20.0f * log10f(lin + 1e-10f);
}

/**
 * main - DSP demonstration
 */
int main(void)
{
  printf("Digital Signal Processing Library\n");
  printf("==================================\n\n");

  /* FFT example */
  printf("FFT Example:\n");
  complex_f32_t signal[8] = {
    {1.0f, 0.0f}, {0.5f, 0.0f}, {0.25f, 0.0f}, {0.125f, 0.0f},
    {0.0625f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}
  };

  fft_radix2(signal, 8);
  printf("FFT computed for 8 samples\n");

  /* Window function example */
  printf("\nWindow Function Example:\n");
  float hamming[16];
  window_hamming(hamming, 16);
  printf("Hamming window computed for 16 samples\n");

  /* FIR filter example */
  printf("\nFIR Filter Example:\n");
  float fir_coeffs[] = { 0.25f, 0.25f, 0.25f, 0.25f };
  fir_filter_t *fir = fir_create(fir_coeffs, 4);
  if (fir) {
    float test_input = 1.0f;
    float output = fir_process(fir, test_input);
    printf("FIR filter output: %f\n", output);
    fir_free(fir);
  }

  /* IIR filter example */
  printf("\nIIR Filter Example:\n");
  float iir_b[] = { 0.5f, 0.5f };
  float iir_a[] = { 1.0f, -0.5f };
  iir_filter_t *iir = iir_create(iir_b, iir_a, 2);
  if (iir) {
    float test_input = 1.0f;
    float output = iir_process(iir, test_input);
    printf("IIR filter output: %f\n", output);
    iir_free(iir);
  }

  printf("\nDSP library test complete\n");

  return 0;
}
