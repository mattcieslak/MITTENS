# Validation against streamlines

`streamline_ground_truth.py` checks MITTENS against a tractogram. The same
streamlines provide the FODs (track-orientation density on the odf8 sphere)
and the ground truth. Each streamline is walked at the model's hop length from
a random phase, in both directions, and every change of voxel is recorded as a
transition.

The script compares two things:

- **One step.** `P(u -> v)` per voxel is compared with the single- and
  double-ODF probabilities.
- **Two steps.** `P(v | w -> u)`, the next voxel given the voxel the walk came
  from, is compared with the first-order probabilities, which ignore `w`.
  This measures what a memoryless voxel chain misses.

Each prediction is a probability distribution over the 26 neighbors. The
score is the L1 distance to ground truth, `Σ_n |p_model(n) - p_truth(n)|`.
Half of it is the fraction of probability mass sent to the wrong neighbors:

- 0 is perfect;
- 0.2 means about 10% of the mass is misplaced;
- 1.0 means about half;
- 2 means none of the mass lands where it should.

Tables report the median over voxels or states. The noise floor is the L1
you would see even with a perfect model, from finite streamline counts. It
is estimated by comparing ground truth from odd and even streamlines,
halved.

For example, take a voxel entered from its right neighbor, moving left:

| next voxel | streamlines | first-order |
|---|---|---|
| l | 57% | 31% |
| li | 27% | 10% |
| back to r | 0% | 31% |

The first-order probabilities forget the incoming direction. Because the FOD
is symmetric, they send about 30% of the mass straight back to the voxel the
walk came from (29.7% averaged over all scored states, versus 0% in the
streamlines). That is most of their L1 of about 1.

## TRXScan sub-60501 (1M ACT + SIFT2 streamlines, 2 mm, 2026-10-29)

```bash
python validation/streamline_ground_truth.py \
    ~/Library/Caches/trxscan/sub-60501/sub-60501_ses-01_space-ACPC_desc-actsift2_tracks.trx
```

The run takes about 2 minutes on 14 cores, of which MITTENS itself takes a
few seconds.

| | step 1.73 mm (v1) | step 2 mm |
|---|---|---|
| **One step, P(u -> v)** (~29.5k voxels) | | |
| isotropic null | 0.76 | 0.84 |
| single-ODF | 0.19 | 0.19 |
| double-ODF | 0.24 | 0.24 |
| noise floor | 0.09 | 0.09 |
| **Two steps, P(v \| w -> u)** (4-5k states) | | |
| first-order single (ignores `w`) | 1.00 | 1.02 |
| noise floor | 0.09 | 0.08 |

What this shows:

- **Single-step probabilities are close but not at the noise floor.** The
  remaining error comes from what the model leaves out: fibers occupying part
  of a voxel and correlated with direction, and real curvature versus a
  uniform cone.
- **On these idealized FODs, single-ODF beats double-ODF.** This is the
  opposite of the paper's FiberCup result.
- **Chaining first-order probabilities is badly wrong over two steps.**
  Ignoring the previous voxel misplaces about half the mass, mostly by
  stepping straight back. Random walks on the voxel graph (v1's walk
  weighting schemes) inherit this. Shortest paths never revisit a voxel, but
  they can still turn sharply from one move to the next.

Caveats:

- The FODs come from the same streamlines, so they are noise-free and not
  deconvolution-blurred. Real FODs will do worse.
- The streamlines come from a tractography algorithm, so ground truth here
  means consistency with that tractogram, not anatomy.
- The tractogram is unoriented, so the transitions are symmetric. This
  phantom cannot test the asymmetric-ODF sign convention.
