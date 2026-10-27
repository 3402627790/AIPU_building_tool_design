// CPU validation only: the control deliberately materializes a complete T.
// This is not the original AIPUGC baseline and does not measure device DDR.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef ELEMENTS
#error ELEMENTS must be set to the tested static shape size
#endif

extern void kernel(const float *x, const float *b, const float *a, float *y);

__attribute__((noinline)) static void control_add(const float *x,
                                                  const float *b, float *t) {
  for (size_t i = 0; i < ELEMENTS; ++i)
    t[i] = x[i] + b[i];
}
__attribute__((noinline)) static void control_mul(const float *t,
                                                  const float *a, float *y) {
  for (size_t i = 0; i < ELEMENTS; ++i)
    y[i] = t[i] * a[i];
}

static int equal(float x, float y) {
  if (isnan(x) || isnan(y))
    return isnan(x) && isnan(y);
  if (x == 0 && y == 0)
    return !!signbit(x) == !!signbit(y);
  return x == y; // Same FP32 rounding, no fast-math or contraction.
}

int main(void) {
  float *x = malloc(ELEMENTS * sizeof(float));
  float *b = malloc(ELEMENTS * sizeof(float));
  float *a = malloc(ELEMENTS * sizeof(float));
  float *t = malloc(ELEMENTS * sizeof(float));
  float *control = malloc(ELEMENTS * sizeof(float));
  float *storage = malloc((ELEMENTS + 32) * sizeof(float));
  if (!x || !b || !a || !t || !control || !storage)
    return 2;
  float *y = storage + 16;
  uint32_t state = 12345;
  for (size_t i = 0; i < ELEMENTS; ++i) {
    state = state * 1664525u + 1013904223u;
    x[i] = ((int)(state % 10001u) - 5000) / 67.0f;
    state = state * 1664525u + 1013904223u;
    b[i] = ((int)(state % 10001u) - 5000) / 113.0f;
    a[i] = ((int)(i % 199) - 99) / 19.0f;
    y[i] = NAN;
  }
  // Exercise zero, negative zero, infinities and NaN as well as ordinary data.
  if (ELEMENTS >= 7) {
    x[0] = -0.0f;
    b[0] = -0.0f;
    a[0] = 1.0f;
    x[1] = 0.0f;
    b[1] = 0.0f;
    a[1] = -1.0f;
    x[2] = INFINITY;
    b[2] = 1.0f;
    a[2] = -2.0f;
    x[3] = -INFINITY;
    b[3] = INFINITY;
    a[3] = 1.0f;
    x[4] = NAN;
    b[4] = 0.0f;
    a[4] = 1.0f;
    x[5] = 1.25f;
    b[5] = -1.25f;
    a[5] = -3.0f;
    x[6] = 2.0f;
    b[6] = 1.0f;
    a[6] = 0.0f;
  }
  for (size_t i = 0; i < 16; ++i)
    storage[i] = storage[ELEMENTS + 16 + i] = 123456.0f;
  control_add(x, b, t);
  control_mul(t, a, control);
  kernel(x, b, a, y);
  for (size_t i = 0; i < ELEMENTS; ++i) {
    volatile float sum = x[i] + b[i];
#ifdef ADD_ON_RHS
    float reference = a[i] * sum;
#else
    float reference = sum * a[i];
#endif
    if (!equal(y[i], reference) || !equal(y[i], control[i])) {
      fprintf(stderr, "mismatch i=%zu got=%g expected=%g control=%g\n", i, y[i],
              reference, control[i]);
      return 1;
    }
  }
  for (size_t i = 0; i < 16; ++i)
    if (storage[i] != 123456.0f || storage[ELEMENTS + 16 + i] != 123456.0f)
      return 3;
  printf("PASS: %d FP32 elements, reference + nonfused CPU control + output "
         "guards\n",
         ELEMENTS);
  free(x);
  free(b);
  free(a);
  free(t);
  free(control);
  free(storage);
  return 0;
}
