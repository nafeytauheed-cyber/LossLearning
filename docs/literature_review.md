# Literature review and originality assessment

Primary sources checked 26 October 2026. This is a focused review, not an exhaustive
systematic search. Publication performance claims are not treated as comparable
results under our much smaller training protocol.

| Paper | Mechanism / assumptions | Strength | Limitations and relationship |
|---|---|---|---|
| [Focal Loss, Lin et al., 2017](https://arxiv.org/abs/1708.02002) | Differentiable modulation emphasizes difficult examples in dense detection. | Addresses easy-negative domination. | Difficulty can be label error. Our baseline uses gamma=2 without alpha weights. |
| [GCE, Zhang & Sabuncu, 2018](https://papers.nips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf) | Interpolates CE and MAE-like classification losses. | Explicit robustness/optimization tradeoff. | May suppress useful hard examples; this is our underlying loss, not our invention. |
| [Learning to Reweight Examples, Ren et al., 2018](https://arxiv.org/html/1803.09050v3) | Meta-gradients guided by clean validation examples. | Direction-aware evidence. | Requires representative clean validation; its layerwise gradient factorization precedes ours. |
| [Adaptive Robust Loss, Barron, 2019](https://arxiv.org/abs/1701.03077) | Continuous residual-loss shape with probabilistic normalization. | Principled adaptive robustness. | Different objective; adapting loss shape is already established. |
| [Meta-Weight-Net, Shu et al., 2019](https://arxiv.org/html/1902.07379v6) | Learns a loss-to-weight mapping using unbiased meta-data. | Flexible learned weighting. | Meta-data assumptions; our rule is a heuristic controller, not learned meta-optimization. |
| [Dataset Cartography, Swayamdipta et al., 2020](https://arxiv.org/abs/2009.10795) | Confidence and variability over training. | Separates ambiguity from persistent difficulty. | Model-dependent evidence does not certify annotation errors. |
| [TracIn, Pruthi et al., 2020](https://arxiv.org/html/2002.08484v3) | Accumulates gradient dot products across checkpoints. | Practical first-order influence. | Our current-head cosine is not accumulated TracIn or a causal guarantee. |
| [Instance-adaptive training, Jin et al., 2021](https://aclanthology.org/2021.emnlp-main.457.pdf) | Predicts instance-specific robustness; joint and quality-supervised variants. | Direct precedent for instance-specific robust losses. | Strongly weakens broad novelty claims. Only the iterative variant uses auxiliary label-quality supervision. |
| [Co-teaching, Han et al., 2018](https://arxiv.org/abs/1804.06872) | Two networks exchange small-loss subsets. | Reduces single-network feedback. | Early-learning/selection assumptions; minorities may have high losses. Not reproduced here. |
| [DivideMix, Li et al., 2020](https://arxiv.org/abs/2002.07394) | Loss-mixture splitting, peer networks and semi-supervised learning. | Strong noisy-image baseline. | Multi-component training; author code linked in paper, not reproduced here. |
| [ELR, Liu et al., 2020](https://arxiv.org/abs/2007.00151) | Regularizes against temporal prediction targets. | Resists later memorization. | Early predictions can be wrong; our optional plain KL is not ELR. |
| [CMW-Net, Shu et al., 2022](https://arxiv.org/abs/2202.05613) | Class-aware meta-weighting. | Handles inter-class differences. | Learned meta-data assumptions differ; class-aware protection is not new. |
| [ProMeCD, Zhu, ICML 2026](https://proceedings.mlr.press/v306/zhu26am.html) | Abstract describes validation-free proportional-integral control using gradient statistics and directional consistency. | Explicitly distinguishes minority underlearning from noise. | Very close motivation. Abstract verified; full PDF fetch failed. Proofs and implementation not independently audited. |
| [ICGNN, Ju et al., ICML 2026](https://proceedings.mlr.press/v306/ju26c.html) | Graph-diffusion influence contradiction, mixture-based noise detection and neighborhood correction. | Close precedent for influence contradiction. | Graph mechanism differs from head-gradient cosine; author-linked code exists. Abstract-level review. |
| [GFD-EMVC, Li et al., ICML 2026](https://proceedings.mlr.press/v306/li26cr.html) | Gradient outliers plus feature-prototype correction for evolutionary multiview classification. | Combines gradient and geometry evidence. | Different application, but limits broad combination novelty. Abstract-level review. |

## Implementation consequences

Use ordinary GCE/focal baselines under matching architectures and budgets. Keep
linear heads available to isolate adaptation. Compare confidence-only adaptation
and neighbor contradiction without gradient alignment. Report minority recall.
Use training-only teacher references; validation only logs progress. Distinguish
exact factorization from head-only SGD approximation and actual AdamW training.

**Novelty remains unestablished.** The candidate contribution is this particular
reproducible controller linking local alignment, entropy, history and conservative
class thresholds. Combining established ideas does not establish a new algorithmic
principle. Detached GCE equals a particular weighted-CE gradient, and binary head
cosine degenerates to signed feature similarity (see mathematics.md).

## Unreproduced strong baselines

DivideMix and ICGNN provide author-linked code; integrating and validating their
published protocols remains future work. Co-teaching, ELR, Meta-Weight-Net and
ProMeCD are discussed, not represented by simplified substitutes bearing those
names. No state-of-the-art comparison or superiority claim is justified by this
release. Confidence filtering may amplify teacher confirmation bias; equal
quotas cannot recover a class the teacher never predicts confidently.
