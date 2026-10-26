# Measured result tables

Balanced accuracy: percentage mean ± sample standard deviation over seeds 11,22,33. Single-seed images use raw proportions. No test-driven selection.

## Synthetic: 20% symmetric noise

| Dataset | Noise | Method | Balanced_accuracy |
| --- | --- | --- | --- |
| blobs | 20% | adaptive | 98.27 ± 1.66 |
| blobs | 20% | ce | 98.27 ± 1.29 |
| blobs | 20% | full | 98.27 ± 1.67 |
| blobs | 20% | gce | 98.00 ± 2.11 |
| circles | 20% | adaptive | 98.27 ± 1.62 |
| circles | 20% | ce | 99.20 ± 1.06 |
| circles | 20% | full | 98.80 ± 0.40 |
| circles | 20% | gce | 99.07 ± 0.61 |
| disconnected | 20% | adaptive | 99.47 ± 0.61 |
| disconnected | 20% | ce | 99.20 ± 0.69 |
| disconnected | 20% | full | 99.33 ± 0.83 |
| disconnected | 20% | gce | 99.47 ± 0.61 |
| moons | 20% | adaptive | 95.47 ± 1.62 |
| moons | 20% | ce | 96.40 ± 1.44 |
| moons | 20% | full | 95.73 ± 1.22 |
| moons | 20% | gce | 94.13 ± 1.62 |
| overlap | 20% | adaptive | 91.20 ± 5.01 |
| overlap | 20% | ce | 92.80 ± 1.74 |
| overlap | 20% | full | 91.47 ± 5.00 |
| overlap | 20% | gce | 91.20 ± 5.01 |
| rare | 20% | adaptive | 95.93 ± 5.42 |
| rare | 20% | ce | 95.26 ± 5.15 |
| rare | 20% | full | 91.26 ± 8.10 |
| rare | 20% | gce | 87.26 ± 2.96 |

## Ablations

| Dataset | Noise | Method | Balanced_accuracy |
| --- | --- | --- | --- |
| moons | 20% | cached | 95.33 ± 1.51 |
| moons | 20% | class_quantile | 96.13 ± 0.46 |
| moons | 20% | confidence_only | 95.20 ± 1.44 |
| moons | 20% | fixed_gce | 96.13 ± 0.61 |
| moons | 20% | full | 95.73 ± 1.22 |
| moons | 20% | neighbors_24 | 95.47 ± 1.62 |
| moons | 20% | neighbors_4 | 94.40 ± 1.20 |
| moons | 20% | no_adaptation | 96.40 ± 0.69 |
| moons | 20% | no_class_safeguard | 95.20 ± 1.74 |
| moons | 20% | no_entropy | 95.73 ± 0.92 |
| moons | 20% | no_gradient | 95.87 ± 1.22 |
| moons | 20% | no_history | 95.73 ± 0.46 |
| moons | 20% | no_potential | 95.47 ± 1.62 |
| moons | 20% | no_stability | 95.73 ± 1.22 |
| rare | 20% | cached | 91.26 ± 8.10 |
| rare | 20% | class_quantile | 99.26 ± 1.28 |
| rare | 20% | confidence_only | 90.59 ± 7.04 |
| rare | 20% | fixed_gce | 87.26 ± 2.96 |
| rare | 20% | full | 91.26 ± 8.10 |
| rare | 20% | neighbors_24 | 90.59 ± 7.04 |
| rare | 20% | neighbors_4 | 91.26 ± 8.10 |
| rare | 20% | no_adaptation | 96.59 ± 3.15 |
| rare | 20% | no_class_safeguard | 90.59 ± 7.04 |
| rare | 20% | no_entropy | 91.26 ± 8.10 |
| rare | 20% | no_gradient | 90.59 ± 7.04 |
| rare | 20% | no_history | 90.59 ± 7.04 |
| rare | 20% | no_potential | 95.93 ± 5.42 |
| rare | 20% | no_stability | 91.26 ± 8.10 |

## Class-dependent noise

| Dataset | Noise | Method | Balanced_accuracy |
| --- | --- | --- | --- |
| blobs | 20% | ce | 98.53 ± 1.51 |
| blobs | 20% | full | 98.13 ± 1.62 |
| blobs | 20% | gce | 98.26 ± 1.41 |
| blobs | 40% | ce | 94.25 ± 4.40 |
| blobs | 40% | full | 95.85 ± 6.15 |
| blobs | 40% | gce | 95.05 ± 6.22 |
| rare | 20% | ce | 95.78 ± 4.12 |
| rare | 20% | full | 99.85 ± 0.26 |
| rare | 20% | gce | 88.59 ± 4.13 |
| rare | 40% | ce | 91.19 ± 1.05 |
| rare | 40% | full | 85.85 ± 8.97 |
| rare | 40% | gce | 81.33 ± 4.16 |

## Reduced Fashion-MNIST

| method | accuracy | balanced_accuracy | mean_q | fallback_fraction |
| --- | --- | --- | --- | --- |
| ce | 0.767 | 0.767 | 0.0 | 1.0 |
| gce | 0.766 | 0.766 | 0.7 | 1.0 |
| adaptive | 0.767 | 0.767 | 0.0 | 1.0 |
| full | 0.773 | 0.773 | 0.0 | 1.0 |

## Paired descriptive statistics

```json
{
  "full_vs_ce_wins": 25,
  "ties": 17,
  "losses": 30,
  "mean_balanced_accuracy_difference_pp": -0.31236745925500914,
  "median_full_to_ce_runtime": 2.3434606109439757
}
```
