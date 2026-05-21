# Node Attributes Tuning Tools

## Purpose
Assume we are doing addition towards two tensors A and B, the result can be represented as Y. All of them are float values, then we have
$$
Y_F = A_F + B_F
$$
With quantization formula $X_F = (X_Q + Z_X)\div S_X$\
We have
$$
\frac{Y_Q+Z_Y}{S_Y} = \frac{A_Q+Z_A}{S_A} + \frac{B_Q+Z_B}{S_B}
$$
After simplification
$$
Y_Q = ([A_Q+Z_A]S_B + [B_Q+Z_B]S_A)\frac{S_Y}{S_AS_B} - Z_Y
$$
