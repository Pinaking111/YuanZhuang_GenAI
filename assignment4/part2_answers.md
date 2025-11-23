# Assignment 4 — Theory answers

## Part 2: Diffusion model questions

### Question 1 — Sinusoidal time embedding formula
For a timestep t and embedding dimension D, the sinusoidal embedding for the i-th pair (0-indexed) is defined as:

- angle = t / (10000^(2i / D))
- emb[2i]   = sin(angle)
- emb[2i+1] = cos(angle)

More explicitly, for dimension index j (0 <= j < D):

\[\text{emb}_j(t) = \begin{cases}
\sin\left( t / 10000^{2i/D} \right) & j = 2i \\
\cos\left( t / 10000^{2i/D} \right) & j = 2i+1
\end{cases}\]

### Question 2 — D = 8, t = 1, max period = 10000
Here D = 8 so i ranges 0..3. The angle for each i is:
- i=0: angle = 1 / 10000^{0} = 1
- i=1: angle = 1 / 10000^{1/4} = 1/10 = 0.1
- i=2: angle = 1 / 10000^{2/4} = 1/100 = 0.01
- i=3: angle = 1 / 10000^{3/4} = 1/1000 = 0.001

So embedding vector (approximate numeric values):

[ sin(1), cos(1), sin(0.1), cos(0.1), sin(0.01), cos(0.01), sin(0.001), cos(0.001) ]

Numerical approximations:
- sin(1) ≈ 0.84147098
- cos(1) ≈ 0.54030231
- sin(0.1) ≈ 0.09983342
- cos(0.1) ≈ 0.99500417
- sin(0.01) ≈ 0.0099998333
- cos(0.01) ≈ 0.99995
- sin(0.001) ≈ 0.0009999998333333417
- cos(0.001) ≈ 0.9999995

So the vector is approximately:

[0.841471, 0.540302, 0.0998334, 0.995004, 0.00999983, 0.99995, 0.00100000, 0.9999995]

### Question 3 — Relation to positional encoding
Both sinusoidal time embeddings (in diffusion models) and positional encodings (in transformers) use the same mathematical construction (sines and cosines at different frequencies) to convert a scalar index (time or position) into a vector representation. The key differences:

- Purpose: positional encodings provide location information for tokens so the transformer can reason about order; time embeddings condition the model on the diffusion timestep (which controls noise level).
- Usage: positional encodings are often added directly to token embeddings before attention layers. Time embeddings are typically passed through an MLP and used as conditioning inputs inside the model (e.g., added to features in residual blocks or used to scale/shift activations).
- Learnable vs fixed: both can be fixed sinusoidal; transformer variants sometimes learn positional embeddings while diffusion code also sometimes uses learned time embeddings.

### Question 4 — Spatial resolution at bottleneck
Input image: 64x64. After three downsampling blocks with stride-2 convolutions, spatial resolution becomes:

64 -> 32 -> 16 -> 8

So the bottleneck resolution is 8 x 8.

### Question 5 — UNet output and loss
In diffusion models a UNet usually predicts one of:
- predicted noise \(\epsilon_\theta(x_t, t)\), or
- predicted clean image \(x_0\), or
- predicted velocity \(v\).

The common training objective is mean-squared error between predicted noise and the true noise used to produce \(x_t\):

\[ L(\theta) = \mathbb{E}_{x_0, \epsilon, t} \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2. \]

When the model predicts \(x_0\) instead, a different reconstruction loss is used but mathematically related to the noise prediction objective.

## Part 3: Energy-based model questions

### Question 6 — Basic gradient calculations
Given \(y = x^2 + 3x\).

a) The derivative w.r.t. x is:
\[ \frac{dy}{dx} = 2x + 3. \]
At x = 2, gradient = 2*2 + 3 = 7.

b) If `requires_grad = False` on x, PyTorch will not track operations on x and `x.grad` will remain `None` after calling `y.backward()`; gradients are not recorded.

c) By default, `torch.tensor(...)` creates a tensor with `requires_grad=False`. Gradients will not be tracked unless you explicitly set `requires_grad=True` or use `torch.tensor(...).requires_grad_()` or create tensors via operations from tensors that require gradients.

Example code and expected output:

```python
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()
print('x.grad =', x.grad)  # => tensor([7.])
```

### Question 7 — Introduce weights
Given:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0])
y = w[0] * x**2 + w[1] * x
```

a) If `w` is created without `requires_grad=True`, then `w.grad` is `None` after `y.backward()`. Only tensors with `requires_grad=True` accumulate gradients.

b) To get gradients w.r.t. `w`, create it with `requires_grad=True`:

```python
import torch
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
# y is a 1-element tensor; backward computes gradients
y.backward()
print('x.grad =', x.grad)  # grads w.r.t x
print('w.grad =', w.grad)  # grads w.r.t w
```

Analytically, \(\partial y/\partial w = [x^2, x]\). With x=2, gradients are [4, 2].

c) As in Question 6c, `torch.tensor` without the flag gives `requires_grad=False` by default.

### Question 8 — Breaking the graph
Code that fails:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.detach()
w = z * 2
w.backward()  # error
```

Why: `y.detach()` creates a new tensor `z` that is disconnected from the computation graph, so `w` does not depend on `x` in the autograd graph, and `backward()` cannot compute gradients w.r.t. `x`.

Fix (one option): re-enable gradient tracking on `z` while keeping its value:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.detach().requires_grad_()
w = z * 2
w.backward()
print('x.grad =', x.grad)  # None, because z was detached
print('z.grad =', z.grad)  # gradients w.r.t z
```

If you want gradients to flow back to x, do not detach z. Simply use `z = y` or avoid detach.

### Question 9 — Gradient accumulation
Example:

```python
x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print('After first backward: x.grad =', x.grad)  # 2
y2 = x * 3
y2.backward()
print('After second backward: x.grad =', x.grad)  # 5 (accumulated)
```

PyTorch accumulates gradients into `.grad`. To avoid accumulation across steps, zero the gradient before each backward pass:

```python
x.grad = None  # or x.grad.zero_()
# or when using optimizers: optimizer.zero_grad()
```

This ensures gradients reflect only the most recent backward call.
