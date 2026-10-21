# Backpropagation derivation

This note describes the derivatives implemented in `numpy_network.py`. Review the chain-rule steps and be ready to explain them in your own words.

## Forward pass

For a batch of \(N\) rows:

\[
Z_1=XW_1+b_1,\quad A_1=\operatorname{ReLU}(Z_1)
\]
\[
Z_2=A_1W_2+b_2,\quad A_2=\operatorname{ReLU}(Z_2)
\]
\[
\hat{Y}=A_2W_3+b_3
\]

The ReLU derivative is 1 when its input is positive and 0 when its input is negative. At exactly zero it is not differentiable; this implementation uses zero there.

## Loss

\[
L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
\]

For one output per example:

\[
\frac{\partial L}{\partial \hat Y}=\frac{2}{N}(\hat Y-Y)
\]

## Output layer

Let \(D_3=\partial L/\partial \hat Y\). Since \(\hat Y=A_2W_3+b_3\):

\[
\frac{\partial L}{\partial W_3}=A_2^\top D_3,\qquad
\frac{\partial L}{\partial b_3}=\sum_{\text{rows}}D_3
\]
\[
D_{A_2}=D_3W_3^\top
\]

## Hidden layers

Apply the ReLU derivative elementwise:

\[
D_2=D_{A_2}\odot\mathbf{1}(Z_2>0)
\]
\[
\frac{\partial L}{\partial W_2}=A_1^\top D_2,\qquad
\frac{\partial L}{\partial b_2}=\sum_{\text{rows}}D_2
\]
\[
D_{A_1}=D_2W_2^\top,\qquad
D_1=D_{A_1}\odot\mathbf{1}(Z_1>0)
\]
\[
\frac{\partial L}{\partial W_1}=X^\top D_1,\qquad
\frac{\partial L}{\partial b_1}=\sum_{\text{rows}}D_1
\]

The sum over rows is needed because each bias is shared by all examples in the batch.

## Gradient descent

For any parameter \(\theta\) and learning rate \(\eta\):

\[
\theta\leftarrow\theta-\eta\frac{\partial L}{\partial\theta}
\]

The analytical gradients are checked in `gradient_check.py` against central finite differences. Record the measured error in your results; do not invent a value.
