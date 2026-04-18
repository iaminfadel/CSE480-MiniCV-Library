# Milestone 2 — Math & Algorithms Notes

## 1. MRMR Feature Selection
MRMR (Minimum Redundancy Maximum Relevance) selects features $f$ that maximize mutual information $MI(f; y)$ with the target variable $y$, while minimizing pairwise mutual information $MI(f; s)$ with already selected features $s \in S$. The criterion is:
$$ \text{score}(f) = MI(f; y) - \frac{1}{|S|} \sum_{s \in S} MI(f; s) $$

## 2. Numerically Stable Softmax
To prevent overflow, the maximum value in the logits $Z$ is subtracted before exponentiation:
$$ P_i = \frac{\exp(Z_i - \max(Z))}{\sum_j \exp(Z_j - \max(Z))} $$

## 3. Cross-Entropy Loss with $\epsilon$ Clipping
Clipping $P$ by adding $\epsilon$ prevents $\log(0)$ errors during backpropagation:
$$ L_{CE} = -\frac{1}{N} \sum_{i=1}^N \log(P[i, y_i] + \epsilon) $$

## 4. Backpropagation (Conv2D and Linear)
For a convolution parameter $W$ in layer $l$, the gradient is computed via transposed convolution (einsum over batches):
$$ \frac{\partial L}{\partial W} = X_{cols}^T \frac{\partial L}{\partial Z} $$
For pooling, error is backpropagated only to the unit that achieved the maximum value inside its receptive field.

## 5. Adam Optimizer
Adam maintains moving averages of the first ($m$) and second ($v$) moments of the gradients, applying bias correction before scaling the update by $\frac{1}{\sqrt{\hat{v}} + \epsilon}$.
$$ m \leftarrow \beta_1 m + (1-\beta_1)g $$
$$ v \leftarrow \beta_2 v + (1-\beta_2)g^2 $$
$$ w \leftarrow w - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} $$

## 6. L2 Regularization & Gradient Clipping
L2 penalizes large weights: $ L = L_0 + \frac{\lambda}{2} \|W\|^2 $. Gradients are updated by adding $\lambda W$.
Gradient clipping caps the global norm $\|g\|$ if it exceeds a threshold $M$:
$$ g \leftarrow g \frac{M}{\max(\|g\|, M)} $$

## 7. Metrics
- **Precision**: $TP / (TP + FP)$
- **Recall**: $TP / (TP + FN)$
- **F1**: Harmonic mean of Precision and Recall.
- **Macro-F1**: Unweighted arithmetic mean of F1 over all classes.
