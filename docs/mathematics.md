# Mathematical specification

## Motivation and assumptions

CE is the negative categorical log likelihood, $-\log p_y$, with logit gradient
$p-e_y$. Its strong correction for improbable observed labels is useful for
learning boundaries but also encourages memorization of annotation errors.
High loss alone cannot separate a true hard example from a mislabeled example.
No unsupervised noise detector can solve that ambiguity without assumptions.
LossLearning assumes a learned neighborhood and an EMA teacher can provide useful,
fallible evidence. It never treats teacher agreement as ground truth.

## Representation

The MLP has a ReLU hidden layer followed by an embedding projection. The CNN has
two convolutions, pooling, and an embedding projection. Both normalize
$h=f_\theta(x)/\max(\|f_\theta(x)\|,10^{-8})$ to limit scale drift.
For each potential center:

$$s_m=\operatorname{softplus}(a_m)+0.05,\quad
p_m=\operatorname{softplus}(b_m)+0.05,$$
$$d_m=\|h-\mu_m\|^2,\qquad \phi_m=\exp[-p_m\log(1+d_m/s_m^2)].$$

These are inverse-multiquadric/RBF-like features, not a new physical law.
The residual linear path remains present:

$$\psi=[h,\phi_1,\ldots,\phi_M,1],\quad z=V\psi,\quad P=\operatorname{softmax}(z).$$

Without potentials, $\psi=[h,1]$. The explicit constant includes the bias in head
gradients. Centers are trainable and need not be unit-normalized. The derivatives
are $\partial\phi/\partial d=-p\phi/(s^2+d)$,
$\partial\phi/\partial h=-2p\phi(h-\mu)/(s^2+d)$,
$\partial\phi/\partial\mu=-\partial\phi/\partial h$,
$\partial\phi/\partial s=2pd\phi/[s(s^2+d)]$, and
$\partial\phi/\partial p=-\phi\log(1+d/s^2)$. Raw parameter derivatives include
the softplus sigmoid chain factor. Joint learning is by autograd.

## Exact final-head gradient factorization

Let $r=P-e_y$. Chain rule gives $\partial\ell/\partial V_{ca}=r_c\psi_a$,
so $g=r\psi^T$. Thus

$$\langle g_i,g_j\rangle_F=\sum_{c,a}r_{ic}r_{jc}\psi_{ia}\psi_{ja}
=(r_i^Tr_j)(\psi_i^T\psi_j).$$

Since $\|r\psi^T\|_F=\|r\|\|\psi\|$, for nonzero norms:

$$\cos(g_i,g_j)=\cos(r_i,r_j)\cos(\psi_i,\psi_j).$$

Clamped norm denominators prevent zero or negligible residuals from generating
artificially strong evidence. This identity is exact for the final linear matrix
V only, at fixed features. Both residuals use the CURRENT STUDENT V, even when
the reference label is a teacher pseudo-label. Using gradients from two differently
parameterized networks would not implement this calculation.

## First-order loss transfer and its scope

For a hypothetical head-only SGD step $V'=V-\eta g_i$, Taylor expansion gives

$$\ell_j(V')-\ell_j(V)=-\eta\langle g_j,g_i\rangle+O(\eta^2\|g_i\|^2).$$

Positive alignment suggests immediate loss reduction for a small enough step.
Cosine removes gradient magnitude; it cannot predict the amount of improvement.
Training uses AdamW on the full network, while this counterfactual excludes
momentum, preconditioning, weight decay, encoder updates and future representation
changes. It is a heuristic feature, not a prediction of the actual AdamW step or
of final generalization. `losslearning-influence` measures explicit autograd
agreement and actual reference CE changes at multiple head-only step sizes.

## Teacher and training-only reference memory

After optimizer steps, $\theta_T\leftarrow\beta\theta_T+(1-\beta)\theta_S$.
Refresh confidence is the teacher's largest class probability. Stability is a .5
EMA of agreement with the preceding refresh's predicted class, initialized at
zero; at least two observations are required. With the default .6 threshold,
three consistently predicted refreshes are needed to qualify.

References are the highest-confidence eligible samples within equal per-predicted-
class quotas $\lfloor R/K\rfloor$. Missing classes remain missing; quotas never
admit low-confidence candidates. The bank keeps IDs, duplicate groups, raw input
tensors, cached student embeddings/head features, pseudo-labels, confidence,
stability, class counts and last refresh step. It scans training data only.
Exact duplicate tabular inputs share group IDs. Image loaders exclude self IDs
but do not globally deduplicate images.

Exact mode recomputes the bounded bank's features with the current student before
neighbor retrieval. Cached mode reuses refresh-time features and CURRENT V:
the identity is then exact for a stale-feature surrogate, not the current model.
`stale_reference.json` measures differences in contradiction scores under both
modes at the same endpoint; paired cached/exact training ablations test outcomes.

## Neighborhood contradiction

Retrieve k cosine-nearest normalized embeddings, exclude self/duplicate IDs,
and softmax similarities divided by temperature to obtain weights $a_{ij}$.
Weights are nonnegative and sum to one if any valid neighbor exists. Invalid
slots have zero weight; an entirely empty row stays zero without NaNs.

$$u_i=\sum_j a_{ij}\cos(g_i,g_j),\qquad
\pi_i(c)=\sum_j a_{ij}\mathbf1[\hat y_j=c],$$
$$H_i=-\sum_c\pi_i(c)\log\pi_i(c),\qquad
C_i=[-u_i]_+\left(1-\frac{H_i}{\log K}\right)\max_{c\ne y_i}\pi_i(c).$$

$K\ge2$ and each factor is in [0,1]. Clamp only the logarithm to implement
$0\log0=0$. Entropy attenuates mixed neighborhoods. Availability requires
min_neighbors valid references AND at least that many references of the observed
class somewhere in the bank. This conservative class-coverage gate is an explicit
addition to the proposal; an unseen rare class is insufficient evidence for
suppression. It can also disable adaptation when the teacher loses a class.

## History and class-aware controller

Live statistics are fixed-size arrays addressed by dense stable training IDs,
not minibatch positions. Scores start at zero and update only with evidence:
$\bar C_i\leftarrow\rho\bar C_i+(1-\rho)C_i$. Missing evidence leaves history
unchanged but forces the current q to zero. A fresh fit resets histories.

$$\tau_c=\operatorname{clip}\left[\tau_0+\delta
(1-\sqrt{n_c/n_{\max}}),0,1\right],$$
$$q_i=q_{\max}r_t m_i\operatorname{clip}[(\bar C_i-\tau_{y_i})/\Delta,0,1].$$

Counts use OBSERVED labels. The ramp is zero during warmup and rises linearly
for ramp_epochs. Underrepresented classes require greater contradiction before
suppression. This is a hypothesis, not a protection theorem: corrupted counts
and teacher bias can defeat it. The quantile ablation raises the threshold to
the class's seen-score 80th percentile when at least ten scores exist, never below
the count-based floor. Suppression budgets are not implemented in v0.1.0.

## Stable adaptive generalized cross-entropy

For detached $q\in[0,1]$:

$$L_q(p_y)=\begin{cases}-\log(p_y)&q=0\\(1-p_y^q)/q&q>0.\end{cases}$$

Set $l=\log p_y$. Since $e^{ql}=1+ql+O(q^2)$,
$\lim_{q\to0}-\operatorname{expm1}(ql)/q=-l$. Log-softmax prevents taking a log
of an underflowed probability. `expm1` avoids cancellation at small positive q;
an explicit safe denominator avoids zero division even in an unselected branch.

$$\frac{\partial L_q}{\partial l}=-e^{ql}=-p_y^q,\quad
\frac{\partial l}{\partial z_k}=\mathbf1[k=y]-p_k,$$
$$\frac{\partial L_q}{\partial z_k}=p_y^q(p_k-\mathbf1[k=y]).$$

At q=0 this is CE. At q=1 the loss $1-p_y$ is proportional to one-hot MAE.
The controller is a detached deterministic rule, not a trainable meta-network.
Differentiating through q would introduce extra terms and change the algorithm.

### Adaptive weighting equivalence

For the present forward pass, define $w_i=\operatorname{stopgrad}(p_{y_i}^{q_i})$.
Then $w_i\mathrm{CE}_i$ and $L_{q_i}$ have identical first-order gradients with
respect to ALL network parameters, although their scalar values differ.
The test suite verifies this. Consequently, changing GCE shape cannot here be
claimed as a fundamentally distinct first-order optimization mechanism from
sample reweighting. The research question concerns the controller's evidence.

### Binary degeneracy and counterexamples

In two-class CE, every nonzero residual lies on one of two opposite rays:
$(-1,1)$ or $(1,-1)$. Residual cosine is therefore +1 for agreeing labels and -1
for disagreeing labels, independently of confidence. The head gradient cosine
reduces to signed feature cosine. For normalized linear-head features,
$\psi_i^T\psi_j=h_i^Th_j+1\ge0$. Moons/circles improvements cannot establish a
unique gradient-influence benefit; multiclass tests and no-gradient controls
are essential.

A confidently wrong homogeneous teacher neighborhood can suppress true minority
examples. Memorized noise can look stable. A rare isolated region may never
qualify for the bank. Entropy/history do not resolve these identifiability problems.

## Objective, cost and differentiability

The minibatch mean GCE optionally adds teacher-to-student KL and a mean-square
parameter penalty. Both coefficients default to zero. AdamW weight decay is
separate and logged. The consistency term is not ELR or DivideMix.

With batch B, bank R, head features D', classes K and neighbors k, similarity
retrieval costs $O(BRD')$, gradient factors $O(Bk(D'+K))$. Similarity chunks bound
pairwise workspace by $O(B\,chunk)$; selected neighbor tensors cost
$O(Bk(D'+K))$. Exact mode additionally recomputes bank embeddings each minibatch.
Raw bank storage is $O(R(input\_size+D'))$; live histories are $O(N)$ and the
bounded diagnostic ring is $O(SN)$, S=40 by default. Epoch logs are $O(T)$.
No full dataset all-pairs matrix or explicit K-by-D' per-example gradient is built.
Float32 training and float64 numerical checks are used. AMP, gradient accumulation,
distributed training and optimizer-aware influence remain future work.
