## Question 1
Given:
- Input size: 8×8
- Kernel size: 4×4
- Stride: 2
- Padding: 1
- Output padding: 1

Using the formula for transposed convolution output size:
output_size = (input_size - 1) * stride + kernel_size - 2 * padding + output_padding

Calculation:
- Width: (8 - 1) * 2 + 4 - 2 * 1 + 1 = 14 + 4 - 2 + 1 = 17
- Height: (8 - 1) * 2 + 4 - 2 * 1 + 1 = 17

Therefore, the output size will be 17×17.

## Question 2
When increasing stride from 2 to 3, keeping all other parameters constant:
- Original output size = (input_size - 1) * 2 + kernel_size - 2 * padding + output_padding
- New output size = (input_size - 1) * 3 + kernel_size - 2 * padding + output_padding

The increase in stride will result in a larger output size because we're increasing the spacing between kernel applications. The difference in output size will be (input_size - 1) pixels for each additional unit of stride.

## Question 3
For a 2D transposed convolution:
output_size = (I - 1) * S + K - 2P + OP

Where:
- I = Input size
- K = Kernel size
- S = Stride
- P = Padding
- OP = Output padding

## Question 4:
To upsample from 16×16 to 32×32 without padding, one possible configuration is:
- Kernel size = 2
- Stride = 2

This works because:
output_size = (16 - 1) * 2 + 2 = 32

## Question 5: 
Given mini-batch [6, 8, 10, 6]:

1. Calculate mean (μ):
   μ = (6 + 8 + 10 + 6) / 4 = 7.5

2. Calculate variance (σ²):
   σ² = [(6-7.5)² + (8-7.5)² + (10-7.5)² + (6-7.5)²] / 4
   = [(-1.5)² + 0.5² + 2.5² + (-1.5)²] / 4
   = (2.25 + 0.25 + 6.25 + 2.25) / 4
   = 11/4 = 2.75

3. Calculate standard deviation (σ):
   σ = √2.75 ≈ 1.658

4. Normalize:
   For each value x: (x - μ) / σ
   
   Result:
   - (6 - 7.5) / 1.658 ≈ -0.904
   - (8 - 7.5) / 1.658 ≈ 0.301
   - (10 - 7.5) / 1.658 ≈ 1.507
   - (6 - 7.5) / 1.658 ≈ -0.904

## Question 6: 
ReLU formula:
f(x) = max(0, x)
```
f(x) = {
    x  if x > 0
    0  if x ≤ 0
}
```

LeakyReLU formula:
f(x) = max(αx, x), where α is a small constant (typically 0.01 or 0.2)
```
f(x) = {
    x      if x > 0
    αx     if x ≤ 0
}
```

## Question 7:
LeakyReLU is often preferred over ReLU in deep networks for several reasons:

1. **Dying ReLU Problem**: ReLU can lead to "dead neurons" where a neuron consistently outputs zero for any input. This happens when a large negative bias leads to the neuron always receiving negative input. LeakyReLU prevents this by allowing a small gradient when the input is negative.

2. **Better Gradient Flow**: The small slope for negative values in LeakyReLU allows for some gradient flow even when the neuron is not active, which can lead to better learning, especially in deep networks.

3. **Non-Zero Centered**: While both activations are non-zero centered, LeakyReLU's small negative values can help with the gradient updates and potentially lead to faster convergence.

4. **Improved Training**: The non-zero gradient for negative values can help with training dynamics and potentially lead to more robust feature learning.