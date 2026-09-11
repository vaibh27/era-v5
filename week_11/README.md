# Week 11 - messing around with Adam, bias correction, warmup, schedules and LR transfer

Support code: `week11_assignment.ipynb`

## What this is

This is my writeup for the week 11 assignment (`assignment.md`): five small experiments on how Adam, bias correction, warmup, and learning-rate schedules behave in practice.

Quick summary of the five things I did:

1. Compute a few steps of Adam by hand for one weight, and check it matches `torch.optim.Adam` exactly.
2. Turn bias correction off and see how much that actually changes the update, and for how long it keeps mattering.
3. Log the update/weight ratio per layer during warmup, to see what it's actually doing.
4. Train the same model with cosine vs WSD schedules, stop both early at step 200, and see which one wins.
5. Sweep the learning rate at a few widths and check if the "best" lr I find at a small width still holds once the model gets wider.

Tasks 1 and 2 are just a single weight tracked by hand, no real training. Tasks 3-5 all use the same small MLP trained on a synthetic regression problem (a random "teacher" network makes up the targets, and a "student" network tries to learn them). I kept the data, batch order, and seed identical across the two runs being compared in each task, so the only thing that's actually different between them is whatever I'm testing - otherwise I wouldn't know if a difference I'm seeing is real or just noise from a different random draw.

## How to run

Everything needed (torch, numpy, matplotlib, jupyter/nbclient) is already installed in the `.venv` folder here. Easiest way to rerun the whole notebook and regenerate every output/plot:

```bash
.venv/bin/python3 -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('week11_assignment.ipynb', as_version=4)
NotebookClient(nb, timeout=1800, kernel_name='python3').execute()
nbformat.write(nb, 'week11_assignment.ipynb')
"
```

or just open it in Jupyter/VS Code and hit Run All. It's all CPU, takes maybe a minute or two - tasks 4 and 5 take the longest since they're sweeping over several learning rates and seeds each.

## What I found, task by task

### Task 1 - doing Adam by hand and checking it against PyTorch

I picked one weight (`w0 = 1.0`) and made up five gradients, then computed `m`, `v`, the bias-corrected `m̂`/`v̂`, and the actual step size by hand, one step at a time. Then I ran the exact same five gradients through real `torch.optim.Adam` and compared every one of those numbers against what's actually sitting inside PyTorch's optimizer state, not just the final weight.

- biggest difference in `m`, `v`, weight: **6.94e-18**
- biggest difference in `m̂`, `v̂`, step size: **5.10e-17**

Both of those are basically floating-point noise, and both are checked with an `assert` in the notebook.

<details>
<summary>Full per-step table (hand vs <code>torch.optim.Adam</code>)</summary>

`w0=1.0`, gradients `[0.1729, -0.37, 0.888, 0.13, -0.23]`, `lr=1e-3`, `β1=0.9`, `β2=0.999`, `eps=1e-8`:

| t | grad | m (hand) | m (torch) | v (hand) | v (torch) | w (hand) | w (torch) | maxdiff |
|---|---|---|---|---|---|---|---|---|
| 1 |  0.17 |  0.0172900 |  0.0172900 |  0.000029894 |  0.000029894 |  0.9990000 |  0.9990000 | 0.0e+00 |
| 2 | -0.37 | -0.0214390 | -0.0214390 |  0.000166765 |  0.000166765 |  0.9993907 |  0.9993907 | 0.0e+00 |
| 3 |  0.89 |  0.0695049 |  0.0695049 |  0.000955142 |  0.000955142 |  0.9989364 |  0.9989364 | 0.0e+00 |
| 4 |  0.13 |  0.0755544 |  0.0755544 |  0.000971087 |  0.000971087 |  0.9984908 |  0.9984908 | 0.0e+00 |
| 5 | -0.23 |  0.0449990 |  0.0449990 |  0.001023016 |  0.001023016 |  0.9982481 |  0.9982481 | 6.9e-18 |

And the bias-corrected versions plus the actual step size (the "torch" step is recovered from the real weight change torch made, not just re-derived from the same formula):

| t | m̂ (hand) | m̂ (torch) | v̂ (hand) | v̂ (torch) | step (hand) | step (torch) | maxdiff |
|---|---|---|---|---|---|---|---|
| 1 |  0.17290000 |  0.17290000 |  0.02989441 |  0.02989441 |  0.00100000 |  0.00100000 | 5.1e-17 |
| 2 | -0.11283684 | -0.11283684 |  0.08342397 |  0.08342397 | -0.00039067 | -0.00039067 | 4.9e-17 |
| 3 |  0.25647565 |  0.25647565 |  0.31869918 |  0.31869918 |  0.00045431 |  0.00045431 | 3.5e-17 |
| 4 |  0.21969878 |  0.21969878 |  0.24313611 |  0.24313611 |  0.00044556 |  0.00044556 | 3.8e-17 |
| 5 |  0.10988491 |  0.10988491 |  0.20501272 |  0.20501272 |  0.00024269 |  0.00024269 | 1.4e-17 |

At t=5: `m̂=0.10988491`, `v̂=0.20501272`, `step=0.00024269`, `w=0.99824811`.

</details>

Honestly I wasn't expecting this to be surprising, and it wasn't - I'm literally just re-implementing the same formula PyTorch already runs internally, so of course it lines up almost exactly. I think the actual point of this task is making sure I understand the formula well enough to reproduce it by hand, not really discovering something new about how Adam behaves.

### Task 2 - what does bias correction actually buy you

Here I used a constant gradient (0.5 every step, no noise) so the only thing changing between the "with correction" run and the "without correction" run is the correction itself, nothing else. At first I just plotted the first 20 steps like the assignment asks, but that turned out to not really be enough to answer the actual question (more on that below).

Without correction, the update is bigger than it should be, by a factor of `(1-β1^t) / sqrt(1-β2^t)`. It's most inflated (~557% too big) around step 12, and after that it decays back down, but really slowly.

**Answer:** I define "stops mattering" as <1% relative difference, giving approximately **3925 steps**. That threshold is a judgment call, not a natural constant - at a looser 10% tolerance it's ~1751 steps, at 5% it's ~2375:

| tolerance | first step it stays below this |
|---|---|
| 10% | ~1751 |
| 5% | ~2375 |
| 1% | ~3925 |

At step 20 (the range the assignment actually asks you to plot) it's still **524%** off, which surprised me a bit.

<details>
<summary>Full first-20-step table (constant gradient g=0.5)</summary>

| step | update, corrected | update, uncorrected | relative diff |
|---|---|---|---|
| 1 | 0.001000 | 0.003162 | 216.23% |
| 2 | 0.001000 | 0.004250 | 324.96% |
| 3 | 0.001000 | 0.004950 | 395.02% |
| 4 | 0.001000 | 0.005442 | 444.16% |
| 5 | 0.001000 | 0.005797 | 479.71% |
| 6 | 0.001000 | 0.006057 | 505.66% |
| 7 | 0.001000 | 0.006245 | 524.49% |
| 8 | 0.001000 | 0.006379 | 537.87% |
| 9 | 0.001000 | 0.006470 | 547.01% |
| 10 | 0.001000 | 0.006528 | 552.79% |
| 11 | 0.001000 | 0.006559 | 555.89% |
| 12 | 0.001000 | 0.006569 | 556.85% |
| 13 | 0.001000 | 0.006561 | 556.09% |
| 14 | 0.001000 | 0.006539 | 553.93% |
| 15 | 0.001000 | 0.006507 | 550.66% |
| 16 | 0.001000 | 0.006465 | 546.49% |
| 17 | 0.001000 | 0.006416 | 541.62% |
| 18 | 0.001000 | 0.006362 | 536.18% |
| 19 | 0.001000 | 0.006303 | 530.30% |
| 20 | 0.001000 | 0.006241 | 524.09% |

The corrected update sits exactly at `lr=0.001` every single step. There's actually a clean reason for that: under a constant gradient, `m̂` works out to exactly `g` and `v̂` works out to exactly `g²` (I worked this out from the EMA recursion - it's a geometric series since the gradient never changes), so the correction cancels out perfectly. I checked this against the simulated numbers with an `assert`.

I originally figured 20 steps would be enough to see the gap close, but it wasn't even close. From what I understand, `β2=0.999` means the second-moment average `v` takes something like `1/(1-β2) = 1000` steps before it really "forgets" that it started at zero - so I reran it out to 6000 steps to actually see where the difference settles down:

| step | relative diff |
|---|---|
| 20 | 524.1% |
| 200 | 134.8% |
| 1000 | 25.8% |
| 2000 | 7.5% |

</details>

So basically, bias correction isn't just a "fixes the first couple steps" thing like I assumed going in - it changes the effective step size for a really long time, thousands of steps in this toy setup. The slow part is driven by `v`, since `β2=0.999` is way closer to 1 than `β1=0.9`, so its bias term takes way longer to fade, and it sits under a square root in the denominator, which drags the whole ratio out.

### Task 3 - update/weight ratio, and what warmup is actually doing

From what I understand, the update-to-weight ratio (`rms(how much a layer moved this step) / rms(the weights themselves)`) is a rough way to check if the optimizer is taking steps that are way too big relative to the weights - if it's close to 1, the step is basically overwriting the layer in one shot, which sounds like a good way to make training unstable. Warmup is supposed to stop that from happening right at the start by slowly ramping the lr up instead of hitting full strength immediately.

I trained the same model twice - once with a 50-step linear warmup, once with no warmup at all - and logged this ratio for every parameter tensor, every step.

At step 1, the ratio between the warmup run and the no-warmup run comes out to exactly **1/50** (the warmup lr scale at step 1) for basically every parameter that doesn't start at zero. LayerNorm biases in this model are zero-initialized though, so their ratio is dividing by ~0 and doesn't mean anything - I excluded those.

<details>
<summary>Full per-parameter table and median-ratio checkpoints</summary>

Step-1 update-to-weight ratio, no warmup vs 50-step warmup (12 params of the `RegressionMLP`, width 256):

| param | no warmup | warmup | ratio |
|---|---|---|---|
| input_layer.weight | 2.95e-02 | 5.90e-04 | 0.0200 |
| input_layer.bias | 2.94e-02 | 5.88e-04 | 0.0200 |
| hidden_layers.0.weight | 8.34e-02 | 1.67e-03 | 0.0200 |
| hidden_layers.0.bias | 8.06e-02 | 1.61e-03 | 0.0200 |
| hidden_layers.1.weight | 8.31e-02 | 1.66e-03 | 0.0200 |
| hidden_layers.1.bias | 8.08e-02 | 1.62e-03 | 0.0200 |
| layer_norms.0.weight | 3.00e-03 | 6.00e-05 | 0.0200 |
| layer_norms.0.bias | excluded (initial weight RMS = 0) | - | - |
| layer_norms.1.weight | 3.00e-03 | 6.00e-05 | 0.0200 |
| layer_norms.1.bias | excluded (initial weight RMS = 0) | - | - |
| output_layer.weight | 8.15e-02 | 1.63e-03 | 0.0200 |
| output_layer.bias | 1.70e-01 | 3.40e-03 | 0.0200 |

Every non-degenerate parameter lands on `0.0200 = 1/50` almost exactly (I loosened the assert tolerance to `1e-3` instead of chasing machine precision, since Adam's `eps` isn't fully negligible for small-gradient params like the LayerNorm weight).

Median warm/no-warm ratio across weight layers, at a few checkpoints:

| step | median ratio |
|---|---|
| 1 | 0.020 |
| 10 | 0.362 |
| 25 | 1.143 |
| 50 | 0.966 |
| 60 | 1.117 |
| 150 | 1.129 |

Last step where the median ratio was still outside `[0.75, 1.33]`: **112**.

</details>

These are two different things. Warmup itself - the lr schedule - is done ramping at **step 50**, full stop, that's just how it's coded; nothing about warmup is still happening after that point. Step 112 is a separate, derived number: it's where a specific diagnostic (median warmup/no-warmup update-ratio landing inside `[0.75, 1.33]`) happens to cross its threshold in this run. That diagnostic is measuring how long it takes the *two training trajectories* (which diverged from each other during the 50 mismatched steps) to look similar again on this ratio - not how long warmup lasts. A plausible contributor is Adam's `m`/`v` running averages needing extra steps to recover after 50 steps of very different update sizes between the runs, but that's a guess about the trajectory-convergence number, not a property of warmup.

**Final answer:** Warmup ends at step 50. The empirical warmup/no-warmup ratio convergence occurs around step 112 under the chosen ±33% criterion, but this is trajectory convergence under that diagnostic, not warmup remaining active until step 112.

### Task 4 - cosine vs WSD, stopped early at step 200

The idea here is to compare two learning-rate schedules - cosine (ramps up, then smoothly decays to ~0) and WSD / warmup-stable-decay (ramps up, stays flat at peak lr for most of training, then decays fast right at the end) - and see which one is better if you have to stop training early, at step 200 out of a planned 300. This felt like a more realistic scenario than always training to completion.

The assignment's own warning ("tune both sides before accepting a comparison") turned out to matter a lot here. My first attempt at picking a shared peak lr for both schedules used a quick 100-step probe, and it gave a wrong-feeling answer - when I looked closer I realized the schedule functions ignore whatever `total_steps` gets passed in and always use their real 300-step shape internally. So a "100-step probe" wasn't actually a rescaled short version of the schedule, it was just the first 100 steps of the real 300-step one - and cosine is already partway decayed by step 100 while WSD (whose decay doesn't start until step 240) is still completely flat. The two schedules were never in a comparable state during that probe, so whichever one "won" wasn't a fair result.

I fixed this by tuning each schedule separately, on its actual full 300-step run, picking whichever peak lr gives the best mean eval loss at step 200 (since that's the actual checkpoint I care about), averaged over 3 seeds instead of 1 so the winning lr isn't just a lucky draw.

**Update after a second pass:** the eval set inside `train_with_schedule` (`features[:512]`) was drawn from the same pool of rows the training batches were sampled from - the same train/eval overlap bug I'd already caught and fixed in Task 5, but hadn't fixed here yet when I first wrote this section. I've now fixed it the same way: training batches are sampled only from the other 1536 rows, and the 512 eval rows are never touched during training. The numbers below are from the corrected, non-leaking version.

<details>
<summary>Full peak-lr sweep (6 candidates × 3 seeds, each on its own full 300-step run, non-leaking eval)</summary>

| schedule | peak lr | eval loss @ step 200 |
|---|---|---|
| cosine | 3e-4 | 7.9663 ± 0.0780 |
| cosine | 1e-3 | 3.1641 ± 0.1195 |
| cosine | 2e-3 | 2.3977 ± 0.0714 |
| cosine | 3e-3 | 2.2916 ± 0.0658 |
| cosine | **5e-3** | **2.1956 ± 0.0730** ← best |
| cosine | 1e-2 | 2.3856 ± 0.0605 |
| WSD | 3e-4 | 6.9736 ± 0.0711 |
| WSD | 1e-3 | 2.6868 ± 0.0974 |
| WSD | 2e-3 | 2.3988 ± 0.0705 |
| WSD | **3e-3** | **2.2804 ± 0.0990** ← best |
| WSD | 5e-3 | 2.3124 ± 0.1027 |
| WSD | 1e-2 | 2.6534 ± 0.0863 |

With each schedule run at its own best peak lr, over 3 seeds:

| checkpoint | metric | cosine (peak=5e-3) | WSD (peak=3e-3) |
|---|---|---|---|
| step 200 | batch loss | 0.2373 ± 0.0234 | 0.3655 ± 0.0085 |
| step 200 | eval loss | 2.1956 ± 0.0730 | 2.2804 ± 0.0990 |
| step 300 | batch loss | 0.1638 ± 0.0159 | 0.0779 ± 0.0043 |
| step 300 | eval loss | 2.1156 ± 0.0710 | 2.0178 ± 0.0788 |

Since cosine and WSD share the same seed (same model init, same batch order), the per-seed paired difference is more informative than comparing means alone - it isolates the schedule effect instead of mixing it with run-to-run variance:

| seed | cosine eval @ 200 | WSD eval @ 200 | cosine − WSD |
|---|---|---|---|
| 0 | 2.2284 | 2.3889 | −0.1605 |
| 1 | 2.0945 | 2.3027 | −0.2082 |
| 2 | 2.2639 | 2.1497 | +0.1143 |

Mean paired diff: −0.0848 ± 0.1421 (n=3). One of the three seeds (seed 2) actually flips - WSD beats cosine on that seed - and the paired standard deviation is larger than the mean difference itself. That's about as direct a demonstration as I could ask for that this "win" isn't robust at n=3.

</details>

**So, which one would I keep?** By the single criterion the assignment asks for - lower mean eval loss at the required step-200 checkpoint - it's still **cosine** (2.1956 vs 2.2804). Because the step-200 evaluation set was also used to select the peak LR, this comparison should be interpreted as a validation-style result rather than a fully held-out test estimate. One thing worth noting: before the eval-set fix, this gap looked like a clean win (0.4255 vs 0.5706, "3-4x the noise"). After fixing the leak, the gap shrank to 0.0848, which is *smaller than either run's own standard deviation* (0.073 and 0.099), and the per-seed breakdown above shows it isn't even one-directional. So with only 3 seeds, I can no longer confidently call this a real effect rather than noise - the direction is the same, but the evidence for it is much weaker than I originally reported. The batch-loss numbers at step 200 (0.2373 ± 0.0234 vs 0.3655 ± 0.0085) are more clearly separated than the eval numbers, so if anything cosine still looks better on the metric it was directly optimizing, just not convincingly so on held-out data.

One scope note: I tune the peak lr for each schedule, but I kept WSD's decay-start fixed at step 240 (80% of the budget) since that's a pretty standard convention, rather than treating it as its own thing to sweep. So this is really a peak-lr comparison under one fixed WSD shape, not a full search over every WSD setting.

And here's the more interesting bit, still holding up after the fix: if you let both schedules run the full 300 steps instead of stopping at 200, WSD wins on eval loss (2.0178 vs 2.1156) - it stayed flat longer at a lower peak lr and then dropped fast at the very end. So the honest answer really depends on which question you're asking: **at step 200, a slight edge to cosine that I'm not fully confident in; if you can afford to run the whole 300 steps, WSD comes out ahead.** That flip (and the fact that the step-200 "win" got a lot less convincing once I fixed the eval leak) is probably the more useful takeaway from this task than either number by itself.

Beyond just this exercise - if I were picking a default for real training, I'd still lean towards WSD, not cosine, mostly for the practical reason rather than the step-200 numbers above: cosine needs you to commit to the total step count upfront, since its whole decay curve is shaped around "when do I hit zero," so stopping early or training longer than planned kind of breaks it. WSD's flat phase doesn't care how long you run it for, so you can checkpoint and decide to decay whenever, which seems more useful in practice than a few-percent edge on one specific loss number that may not even be distinguishable from noise.

### Task 5 - does the best learning rate carry over to a wider model

Here I trained the same architecture at three widths (256, 512, 1024), swept the learning rate at each one, and tried to see if there's a pattern I could use to guess a good lr for a much wider model (4096) without actually training one.

| width | best lr | mean eval MSE (±std) |
|---|---|---|
| 256 | 3e-3 | 1.9935 ± 0.0389 |
| 512 | 3e-3 | 1.7954 ± 0.0109 |
| 1024 | 1e-3 | 1.7203 ± 0.0429 |

So the best lr doesn't just stay put - it's the same at 256 and 512, but drops once we get to 1024. Since the three points didn't all agree, I used a log-width/log-LR fit (`np.polyfit` over all three points) as a heuristic extrapolation out to 4096, snapped to the nearest lr I actually tested. Because it is based on only three discrete LR optima, I treat **3e-4** as a hypothesis rather than a reliable scaling rule.

<details>
<summary>Full sweep (3 widths × 6 learning rates × 3 seeds, held-out eval)</summary>

| width | lr | eval MSE |
|---|---|---|
| 256 | 3e-5 | 16.8241 ± 0.0573 |
| 256 | 1e-4 | 8.6936 ± 0.0637 |
| 256 | 3e-4 | 4.3308 ± 0.1707 |
| 256 | 1e-3 | 2.1968 ± 0.0726 |
| 256 | **3e-3** | **1.9935 ± 0.0389** ← best |
| 256 | 1e-2 | 2.1677 ± 0.1034 |
| 512 | 3e-5 | 9.2930 ± 0.1508 |
| 512 | 1e-4 | 6.0638 ± 0.1449 |
| 512 | 3e-4 | 2.2319 ± 0.0479 |
| 512 | 1e-3 | 1.8195 ± 0.0169 |
| 512 | **3e-3** | **1.7954 ± 0.0109** ← best |
| 512 | 1e-2 | 2.2263 ± 0.0645 |
| 1024 | 3e-5 | 7.3240 ± 0.0565 |
| 1024 | 1e-4 | 2.6072 ± 0.0701 |
| 1024 | 3e-4 | 1.7898 ± 0.0102 |
| 1024 | **1e-3** | **1.7203 ± 0.0429** ← best |
| 1024 | 3e-3 | 1.8954 ± 0.0280 |
| 1024 | 1e-2 | 2.2114 ± 0.0616 |

The fit comes out to about **-0.239 decades of lr per width-doubling**; plugging width 4096 into it gives a raw value of `4.00e-4`, snapped to `3e-4` since that's a grid point I actually tested.

</details>

Quick note on how I got these numbers, because it wasn't right the first time: originally I measured `final_loss` on `features[:512]`, but the training batches were being drawn from the full 2048-row set - so the "held-out" rows weren't actually held out, they were also being trained on. That's not really testing generalization, it's just testing how well each lr fits the training data it already saw. I fixed it by splitting the data properly: training only ever samples from the other 1536 rows, and the 512 eval rows are never touched during training. The absolute loss numbers went up a lot after the fix (from roughly 0.03-0.1 to roughly 1.7-2.2), which makes sense - that's just the real generalization gap on a smaller training set showing up once I stopped accidentally cheating. The best-lr trend and the final width-4096 guess happened to come out the same either way, but I think that's mostly a coincidence, not something I'd have counted on.

**How confident am I in the 4096 number?** Medium-low, honestly. This is extrapolating a full width-doubling past anything I actually tested - 1024 to 4096 is a big jump, and I'm only fitting through 3 points. If all three widths had landed on the same best lr, I'd trust the extrapolation a lot more; since they didn't, this really is a first guess, not something I'd stake much on without actually running 4096 to check.

## Bugs I ran into while doing this (and how I found them)

I went back over this notebook a couple times, and both passes turned up real mistakes:

- **Task 4's first lr probe wasn't actually fair.** I mentioned this above, but the short version: I tried to save time with a 100-step probe to pick one shared peak lr, but the schedule functions secretly always use their real 300-step shape regardless of what I passed in, so the probe wasn't testing what I thought it was testing. Fixed by tuning each schedule on its real, full-length run instead.
- **Task 5's eval set was leaking into training.** The "held-out" 512 rows were also reachable by the training batch sampler, so it wasn't measuring generalization at all. Fixed with an actual train/eval split.
- **Task 4's eval set had the same leak as Task 5.** Same bug, different task - I caught and fixed Task 5's version first, then noticed the identical leak in `train_with_schedule` and fixed it the same way (train batches only from the other 1536 rows, eval rows untouched). This mattered more than I expected: the leaky version made cosine's step-200 win look like "3-4x the noise," but with the leak fixed the gap shrank to *smaller* than either run's own std, so the real takeaway is "weak edge to cosine, not confidently distinguishable from noise at n=3 seeds" rather than a clean win.
- **Task 3's LayerNorm bias values were meaningless.** They start at exactly 0, so update/weight was dividing by ~0 and blowing up to something like 1e9 in the plots and tables. I excluded them and noted why.

The assignment's own warning ("tune both sides before accepting a comparison") is basically what went wrong in my first pass at Task 4.

## Things I know are still limited about this

- WSD's decay-start point (80% of the training budget) is just a fixed convention I used, not something I actually tuned - a real comparison might tune that too.
- The width-4096 lr guess in Task 5 is an extrapolation past anything I tested, from only 3 data points, so I'm treating it as a first guess rather than something I'd bet on.
- Task 4's step-200 cosine-vs-WSD comparison is only 3 seeds, and after fixing the eval leak the gap is close to that noise level - I wouldn't treat "cosine wins at step 200" as a strong claim, just a weak lean that happens to point the same direction as before the fix.
