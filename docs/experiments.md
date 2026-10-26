# Experiment interpretation

All complete tables are generated from saved metrics in [result_tables.md](result_tables.md).
The release includes 504 symmetric-noise synthetic fits, 84 ablations, 36 class-
dependent fits, 2 quick demonstration fits and 4 reduced Fashion-MNIST fits.
The audit checked all 626 synthetic fit outputs against saved predictions, split
IDs, corruption masks and final q values. Neither validation nor test data enters
the reference bank. Validation logs progress only; no model selection was performed.

## Primary conclusion

The mechanism is implemented correctly, but its performance hypothesis is not
established. Across 72 matched synthetic dataset/noise/seed settings, full versus
CE has 25 wins, 17 ties and 30 losses. Mean balanced-accuracy difference is -0.3124
percentage points. Small differences are not evidence of statistical significance.
There is no state-of-the-art claim and no proof of general noise identification.

At 20% noise, moons and overlap both favor CE on average. The rare-task full model
loses 4 balanced-accuracy points relative to CE. At 10% noise on the rare task it
improves by 1.33 points, but clean rare data loses 4 points. This variability is
why the favorable seed-42 quick result (94.4% versus 97.6%) is not the headline
claim. See per-class recall, not merely majority-dominated overall accuracy.

## Which components helped?

No component was uniformly beneficial. At 20% noise on moons, the full model is
95.73%, no-gradient 95.87%, no-history 95.73%, no-entropy 95.73%, confidence-only
95.20%, and same-head CE 96.40% balanced accuracy. These do not establish unique
value from the gradient signal, history or entropy.

On the rare task, full is 91.26%, no-gradient 90.59%, no-history 90.59%, no-class-
safeguard 90.59%, and same-head CE 96.59%. Conservative count thresholds help
slightly relative to removing protection but fail to protect minority performance
adequately. The class-quantile variant reaches 99.26%, with substantially less
suppression. This is a promising exploratory lead among fourteen variants; it
needs new datasets and untouched evaluation seeds before promotion to a default.
No parameter was changed based on the observed test outcomes.

Linear-head adaptive reaches 95.93% on rare data, better than full's 91.26%;
potentials are not consistently helpful. Wider-MLP CE and same-head CE controls
are included. Cached mode is similar on rare data and somewhat worse on moons;
a one-run stale-feature diagnostic shows small mean but occasionally large maximum
score deviations because neighborhood identity can change.

## Failure modes visible in logs

Many 40%-noise runs end at q=0 because confidence/class coverage is insufficient.
Such fallback preserves CE behavior but prevents the proposed mechanism from
helping. Full-model differences can then arise from architecture or earlier phases,
so compare with potential_ce and inspect q histories before crediting adaptation.
The binary gradient-cosine identity reduces largely to signed geometric agreement.
Three-class blobs provide a multiclass check but are too easy for a strong conclusion.

Noise AUROC and hard-clean suppression statistics are in per-run metrics. The
hard-clean definition is post-training top-quintile observed-label loss among
actually clean training points. It is a model-dependent proxy, not a ground-truth
boundary label. This diagnostic information is never fed into training.

## Images, approximation and cost

Fashion-MNIST used 2,000/1,000/1,000 train/validation/test images, ten epochs and
one seed with 20% training noise. CE 76.7%, GCE 76.6%, adaptive linear 76.7%, full
77.3%. Both adaptive variants ended at complete fallback. This validates the image
pipeline, not an image robustness improvement. CIFAR download was slow and incomplete
when interrupted; training did not run. No full-scale image protocol is claimed.

Head-gradient factorization maximum error is about 7.1e-15 in float64. Relative
Taylor error grows from 1.7e-5 at step 1e-5 to .20 at step .1 and .95 at step 1.
That validates local algebra, not the pseudo-label assumptions or actual AdamW
updates. Stale-reference mean score error spans about .0017 to .0132 in the supplied
endpoint checks; maximum errors are much larger. Details are in influence JSONs.

Full-model median paired runtime is 2.34 times CE on this shared CPU host.
Binary 2D parameter counts: linear 2,338; potential 2,914; wider linear 2,898.
Runtime includes fitting and initialization after lazy optimizer warmup. Some
jobs shared CPU resources, and CPU per-run peak memory was not measured. No
isolated hardware efficiency claim is justified.
