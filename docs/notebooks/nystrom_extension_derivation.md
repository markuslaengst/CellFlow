# Nyström Extension for Graph Laplacian Spectral Embeddings

*A derivation for unnormalized and symmetric normalized graph Laplacians, biharmonic weighting, and gradients.*

## Overview

Both Nyström extension formulas in the notebook follow the same idea: **write the Laplacian eigenvector equation at a new point and solve for the unknown eigenfunction value at that point.**

The current notebook setting is:

```python
kernel_laplacian_type = "unnormalized"
kernel_spectral_weight_type = "biharmonic"
```

Therefore, the **unnormalized formula** is the one currently used. The symmetric normalized formula applies if the notebook instead constructs and solves the eigenproblem for the symmetric normalized Laplacian.

We will derive both formulas, explain the separate spectral weighting, check reproduction at construction points, and differentiate the extensions for use in the vector field.

## 1. Graph and spectral embedding setup

Start with a weighted graph on construction points $x_1,\ldots,x_N$. Define its affinity matrix, vertex degrees, and degree matrix by

$$
W_{ij}=w(x_i,x_j),\qquad
 d_i=\sum_{j=1}^{N}W_{ij},\qquad
D=\operatorname{diag}(d_1,\ldots,d_N).
$$

For each retained spectral mode $\ell$, the eigensolver produces

$$
L\phi_\ell=\lambda_\ell\phi_\ell,
$$

where $\phi_{j\ell}$ denotes the $j$-th component of eigenvector $\phi_\ell$, i.e. its value at construction point $x_j$.

These discrete values initially exist only at the construction points. The **Nyström extension** assigns a value $f_\ell(x)$ to an arbitrary query point $x$ using the fitted eigenvectors and the query affinities

$$
w_j(x)=w(x,x_j),\qquad d(x)=\sum_{j=1}^{N}w_j(x).
$$

The final spectral embedding coordinate is

$$
\boxed{\Psi_\ell(x)=a_\ell f_\ell(x)},
$$

where $a_\ell$ is a **separate spectral weight**, applied after extending the eigenfunction.

## 2. Unnormalized graph Laplacian

### 2.1 Derive the extension

The unnormalized graph Laplacian is

$$
L_{\mathrm{un}}=D-W.
$$

Its eigenvector equation is

$$
(D-W)\phi_\ell=\lambda_\ell\phi_\ell.
$$

Taking row $i$ gives

$$
d_i\phi_{i\ell}-\sum_j W_{ij}\phi_{j\ell}
=\lambda_\ell\phi_{i\ell}.
$$

Move the eigenvalue term to the left:

$$
(d_i-\lambda_\ell)\phi_{i\ell}
=\sum_jW_{ij}\phi_{j\ell}.
$$

Assuming $d_i-\lambda_\ell\neq0$, we can solve for the coordinate at construction point $x_i$:

$$
\phi_{i\ell}
=\frac{\sum_jW_{ij}\phi_{j\ell}}{d_i-\lambda_\ell}.
$$

Apply the same equation at a query point $x$, replacing the graph row with query weights $w_j(x)$:

$$
d(x)f_\ell(x)-\sum_jw_j(x)\phi_{j\ell}
=\lambda_\ell f_\ell(x).
$$

Solving for the unknown value yields

$$
\boxed{
f_\ell(x)=\frac{\sum_jw_j(x)\phi_{j\ell}}{d(x)-\lambda_\ell}
}.
$$

After spectral scaling, the **unnormalized Nyström embedding** is

$$
\boxed{
\Psi_\ell(x)=a_\ell\frac{\sum_jw_j(x)\phi_{j\ell}}{d(x)-\lambda_\ell}
}.
$$

**Interpretation:** The denominator $d(x)-\lambda_\ell$ comes directly from the eigenfunction equation. It is not an independently chosen kernel normalization.

## 3. Symmetric normalized graph Laplacian

### 3.1 Derive the extension

The symmetric normalized graph Laplacian is

$$
L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}.
$$

Assuming positive degrees, its eigenvector equation at construction point $i$ reads

$$
\phi_{i\ell}
-\frac{1}{\sqrt{d_i}}
  \sum_jW_{ij}\frac{\phi_{j\ell}}{\sqrt{d_j}}
=\lambda_\ell\phi_{i\ell}.
$$

Rearranging gives

$$
(1-\lambda_\ell)\phi_{i\ell}
=\frac{1}{\sqrt{d_i}}
  \sum_jW_{ij}\frac{\phi_{j\ell}}{\sqrt{d_j}}.
$$

At a query point, use the query affinities $w_j(x)$ and query degree $d(x)$ while retaining the fitted construction degrees $d_j$:

$$
(1-\lambda_\ell)f_\ell(x)
=\frac{1}{\sqrt{d(x)}}
  \sum_jw_j(x)\frac{\phi_{j\ell}}{\sqrt{d_j}}.
$$

Therefore,

$$
\boxed{
f_\ell(x)=
\frac{\displaystyle\sum_jw_j(x)\phi_{j\ell}/\sqrt{d_j}}
{\sqrt{d(x)}\,(1-\lambda_\ell)}
}.
$$

The corresponding **symmetric normalized Nyström embedding** is

$$
\boxed{
\Psi_\ell(x)=a_\ell
\frac{\displaystyle\sum_jw_j(x)\phi_{j\ell}/\sqrt{d_j}}
{\sqrt{d(x)}\,(1-\lambda_\ell)}
}.
$$

Each factor follows directly from the operator:

- $1/\sqrt{d_j}$ comes from the **right** $D^{-1/2}$.
- $1/\sqrt{d(x)}$ comes from the **left** $D^{-1/2}$.
- $1/(1-\lambda_\ell)$ comes from rearranging the identity term in $L_{\mathrm{sym}}$.

### 3.2 Connection to the standard Nyström formula

Define the normalized affinity matrix

$$
S=D^{-1/2}WD^{-1/2}.
$$

Because $L_{\mathrm{sym}}=I-S$,

$$
S\phi_\ell=(1-\lambda_\ell)\phi_\ell.
$$

Thus, the affinity eigenvalue corresponding to Laplacian eigenvalue $\lambda_\ell$ is

$$
\mu_\ell=1-\lambda_\ell.
$$

The usual Nyström expression divides the weighted eigenvector sum by the **affinity eigenvalue** $\mu_\ell$. This explains why the normalized extension contains $1-\lambda_\ell$, not $\lambda_\ell$, in its denominator. This perspective is also discussed in Mahadevan and Maggioni, Section 6.1.

> **Important:** The two extension formulas use eigenpairs of *different Laplacians*. Switching extension formulas also requires fitting the corresponding graph operator and its eigenvectors; the formulas are not interchangeable for fixed eigenpairs.

## 4. Spectral scaling and biharmonic distance

The Nyström extension provides $f_\ell(x)$. A separate step weights the modes:

$$
\Psi_\ell(x)=a_\ell f_\ell(x).
$$

Under the notebook setting

```python
kernel_spectral_weight_type = "biharmonic"
```

the embedding scale is

$$
\boxed{
a_\ell=\frac{1}{\lambda_\ell+\varepsilon},
\qquad \varepsilon=10^{-8}
}.
$$

With the current **unnormalized** Laplacian, the complete coordinate is therefore

$$
\boxed{
\Psi_\ell(x)
=\frac{\sum_jw_j(x)\phi_{j\ell}}
{(\lambda_\ell+\varepsilon)(d(x)-\lambda_\ell)}
}.
$$

The two denominator factors have different roles:

- **$d(x)-\lambda_\ell$:** arises from extending the eigenfunction for the chosen Laplacian.
- **$\lambda_\ell+\varepsilon$:** weights that mode in the final embedding.

For 32 retained modes, squared Euclidean distance in embedding space becomes

$$
\boxed{
\|\Psi(x)-\Psi(y)\|_2^2
=\sum_{\ell=1}^{32}
\frac{\bigl(f_\ell(x)-f_\ell(y)\bigr)^2}
{(\lambda_\ell+\varepsilon)^2}
}.
$$

This is the **truncated, regularized biharmonic spectral distance** used here. Modes with smaller nonzero Laplacian eigenvalues receive greater weight. The trivial zero mode is typically excluded from the retained distance coordinates.

## 5. Does the extension reproduce values at construction points?

### 5.1 Unnormalized case: self-affinities cancel

There is a subtle point: query evaluation includes the self-affinity $w(x_i,x_i)=1$, whereas the unnormalized graph used for the eigenproblem has no diagonal entries.

The two conventions are consistent with the **unnormalized** Laplacian. Adding self-affinities changes the matrices to

$$
\overline W=W+I,\qquad \overline D=D+I,
$$

but leaves the Laplacian unchanged:

$$
\overline D-\overline W
=(D+I)-(W+I)=D-W.
$$

More explicitly, at a construction point $x_i$ the query degree is

$$
d(x_i)=d_i+1.
$$

Using the original eigenvector equation, the extension numerator becomes

$$
\begin{aligned}
\sum_jw_j(x_i)\phi_{j\ell}
&=\sum_jW_{ij}\phi_{j\ell}+\phi_{i\ell}\\
&=(d_i-\lambda_\ell)\phi_{i\ell}+\phi_{i\ell}\\
&=(d_i+1-\lambda_\ell)\phi_{i\ell}.
\end{aligned}
$$

Dividing by the extension denominator gives

$$
f_\ell(x_i)
=\frac{(d_i+1-\lambda_\ell)\phi_{i\ell}}
{d_i+1-\lambda_\ell}
=\phi_{i\ell},
$$

provided that denominator is nonzero.

### 5.2 Symmetric normalized case: self-affinities matter

For the symmetric normalized Laplacian, adding diagonal self-affinities **changes the normalized operator**, since the degrees inside $D^{-1/2}$ change. The notebook accordingly includes self-affinities when fitting normalized graphs so that construction and query conventions agree.

### 5.3 Consistent local bandwidths

The bandwidth convention also supports consistency:

- **Construction scale:** the distance to the 24th *non-self* neighbor.
- **Query scale:** the distance to the 25th construction point, including a potential exact self-match.

At a construction point, its zero-distance self-match occupies one position, so these neighbor ranks yield the same bandwidth (assuming the expected neighbor ordering/tie handling).

### 5.4 What is held fixed?

The extension keeps the following fitted quantities fixed:

- Eigenvalues and eigenvectors.
- Construction-point degrees.
- Construction-point bandwidths.

It defines a continuation from the **fitted model**. Recomputing the eigendecomposition after inserting $x$ as a new graph vertex would generally change those quantities and need not give the same result.

## 6. Gradients of the Nyström extension

The downstream vector field needs gradients of embedding coordinates with respect to the query point $x$.

### 6.1 Unnormalized case

Define

$$
S_\ell(x)=\sum_jw_j(x)\phi_{j\ell}.
$$

The embedding coordinate is

$$
\Psi_\ell(x)=a_\ell\frac{S_\ell(x)}{d(x)-\lambda_\ell}.
$$

Holding the fitted eigenpairs and $a_\ell$ constant, and applying the quotient rule,

$$
\begin{aligned}
\nabla\Psi_\ell(x)
&=\frac{a_\ell}{d(x)-\lambda_\ell}\nabla S_\ell(x)\\
&\quad-\frac{a_\ell S_\ell(x)}{(d(x)-\lambda_\ell)^2}\nabla d(x).
\end{aligned}
$$

Since

$$
\nabla S_\ell(x)=\sum_j\phi_{j\ell}\nabla w_j(x),
\qquad
\nabla d(x)=\sum_j\nabla w_j(x),
$$

we obtain

$$
\boxed{
\begin{aligned}
\nabla\Psi_\ell(x)
&=\frac{a_\ell}{d(x)-\lambda_\ell}
  \sum_j\phi_{j\ell}\nabla w_j(x)\\
&\quad-\frac{a_\ell S_\ell(x)}{(d(x)-\lambda_\ell)^2}
  \sum_j\nabla w_j(x).
\end{aligned}
}
$$

**Key detail:** The second term is essential because the query degree $d(x)$ changes with $x$. Treating it as constant would omit part of the derivative.

### 6.2 Symmetric normalized case

Define

$$
T_\ell(x)=\sum_jw_j(x)\frac{\phi_{j\ell}}{\sqrt{d_j}}.
$$

Then

$$
\Psi_\ell(x)=\frac{a_\ell}{1-\lambda_\ell}\frac{T_\ell(x)}{\sqrt{d(x)}}.
$$

Differentiating the product $T_\ell(x)d(x)^{-1/2}$ gives

$$
\boxed{
\begin{aligned}
\nabla\Psi_\ell(x)
&=\frac{a_\ell}{(1-\lambda_\ell)\sqrt{d(x)}}
  \sum_j\frac{\phi_{j\ell}}{\sqrt{d_j}}\nabla w_j(x)\\
&\quad-\frac{\Psi_\ell(x)}{2d(x)}
  \sum_j\nabla w_j(x).
\end{aligned}
}
$$

These gradient contributions correspond to the notebook's `component_factors` and `factor_derivatives` in the extension code.

## 7. Differentiating variable-bandwidth kernel weights

For variable-bandwidth kernels, $\nabla w_j(x)$ must also account for changes in the **query bandwidth** $\rho(x)$.

Consider the scaled squared distance

$$
s_j(x)
=\frac{\|x-x_j\|_2^2}
{\delta^2\rho(x)\rho_j},
$$

where $\rho_j$ is the fixed bandwidth associated with construction point $x_j$ and $\delta$ is a global scale.

Within a region where the relevant neighbor ordering stays fixed, the chain rule gives

$$
\boxed{
\nabla s_j(x)
=\frac{2(x-x_j)}{\delta^2\rho(x)\rho_j}
-\frac{s_j(x)}{\rho(x)}\nabla\rho(x)
}.
$$

If $w_j(x)=h(s_j(x))$, then

$$
\boxed{\nabla w_j(x)=h'(s_j(x))\nabla s_j(x)}.
$$

Therefore, differentiating the kernel must include **both** the derivative of the squared distance and the derivative of the query bandwidth. The kernel implementation includes both contributions.

At nearest-neighbor ties or discontinuous support boundaries, ordinary differentiability may fail, so these formulas hold on regions where the relevant quantities are differentiable.

## 8. Numerical considerations and limitations

The derivations assume that all relevant denominators are nonzero and, for normalized Laplacians, that the involved degrees are positive.

In practice:

- **Unnormalized extension:** $d(x)-\lambda_\ell$ can approach zero.
- **Symmetric normalized extension:** $1-\lambda_\ell$ can approach zero.
- **Biharmonic scaling:** $\lambda_\ell+\varepsilon$ regularizes the spectral weighting near zero eigenvalues.
- **Variable-bandwidth kernels:** neighbor ties and support changes can make the extension only piecewise differentiable.

The implementation floors very small extension denominators as a numerical safeguard. **Where that safeguard is active, the effective function is modified**, so the displayed analytic gradients describe the unfloored formulas unless the derivative of the safeguard is explicitly accounted for. Away from such regions, the formulas above agree with the derived extension and gradients.

## 9. Formula summary

| Operator | Nyström eigenfunction extension $f_\ell(x)$ |
|---|---|
| Unnormalized $L_{\mathrm{un}}=D-W$ | $\displaystyle\frac{\sum_j w_j(x)\phi_{j\ell}}{d(x)-\lambda_\ell}$ |
| Symmetric normalized $L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}$ | $\displaystyle\frac{\sum_j w_j(x)\phi_{j\ell}/\sqrt{d_j}}{\sqrt{d(x)}(1-\lambda_\ell)}$ |

In either case, apply the embedding scale separately:

$$
\boxed{\Psi_\ell(x)=a_\ell f_\ell(x),\qquad
 a_\ell=\frac{1}{\lambda_\ell+10^{-8}}\quad\text{(biharmonic setting)}.}
$$

**Takeaway:** The Laplacian eigenvector equation determines *how a mode is extended*, while the spectral factor $a_\ell$ determines *how strongly that mode contributes to the embedding distance*.
