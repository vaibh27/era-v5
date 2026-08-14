# Homomorphic Arithmetic Embeddings — Prototype 01

This repository implements the first computational stage of the research report:

> Can a learned latent arithmetic subspace represent an exact algebraic structure,
> while fixed operators act on that representation as arithmetic operations?

The prototype has three parts.

1. `code/crt.py` is the exact mathematical oracle for the finite/bounded CRT/RNS construction.
2. `code/evaluate_crt.py` exhaustively evaluates that construction and writes a reproducible exact-result artifact.
3. `code/train.py` learns an embedding table plus addition and multiplication operators and evaluates compositional generalization.

## Important scope

This first prototype works over the finite ring `Z_M` (arithmetic modulo `M`). It does **not** claim unbounded integer arithmetic. The report establishes the bounded-domain theorem; neural learnability is the empirical question being tested here.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run the exact CRT baseline

```bash
cd code
python evaluate_crt.py --moduli 5,7,11 --out ../artifacts/crt_exact.json
```

This exhaustively evaluates all 148,225 operand pairs in `Z_385`; it should report
an exact accuracy of `1.0` for decoding, addition, and multiplication. This is the
constructive proof-of-concept for bounded arithmetic, not a learning result.

## Run the continuous neural-vector realization

```bash
cd code
python evaluate_continuous_crt.py --moduli 5,7,11 --out ../artifacts/continuous_crt_exact.json
```

This represents each residue channel as a real one-hot vector and uses fixed
bilinear tensors for the arithmetic operators. With `(5,7,11)`, the arithmetic
subspace is 23-dimensional (`5 + 7 + 11`) and remains exact over `Z_385`. The
`HybridArithmeticEmbedding` class in `code/continuous_crt.py` concatenates this
math subspace with an independent trainable semantic embedding; arithmetic acts
only on the protected math prefix.

For the lower-level theorem check:

```bash
PYTHONPATH=code python -c "from crt import CRTSystem, verify_exhaustive; verify_exhaustive(CRTSystem((5,7,11))); print('CRT verification passed')"
```

## Run the learned baseline

The default experiment uses `Z_101`, a 16-dimensional arithmetic embedding, an MLP addition operator, and a low-rank bilinear multiplication operator.

```bash
cd code
python train.py \
  --modulus 101 \
  --dim 16 \
  --rank 16 \
  --epochs 100 \
  --train-fraction 0.8 \
  --out ../artifacts/run.json
```

The output reports:

- addition exact accuracy by nearest learned embedding,
- multiplication exact accuracy,
- latent MSE for addition,
- latent MSE for multiplication.

## What the experiment actually tests

For a held-out pair `(x, y)`, the model predicts latent vectors:

```text
A+(E(x), E(y))
A*(E(x), E(y))
```

and checks whether they land on:

```text
E((x + y) mod M)
E((x * y) mod M)
```

This is stronger than only checking a decoded answer because it measures the proposed latent homomorphism directly. It is a learnability baseline, not the proof of the bounded-domain theorem; compare it with `crt_exact.json` rather than claiming that a low learned accuracy refutes the construction.

## Next experiments

1. Add a fixed CRT/Fourier baseline.
2. Add associativity and distributivity tests.
3. Sweep arithmetic dimension.
4. Test range extrapolation.
5. Add a semantic branch and test semantic preservation.
6. Compare learned structure against xVal/FoNE/Abacus-style baselines where appropriate.

## Variable-length latent arithmetic: milestone 1

The `src/arithmetic/` package starts the separate variable-length research track.
It uses little-endian base-`B` limbs, not a lookup embedding for whole integers.
`RecurrentLatentAdder` applies one shared `CarryCell` at every limb position and
predicts the result limb, carry-out, and result-limb latent vector at each step.

Run the first multi-limb experiment from this directory:

```bash
./.venv/bin/python train_addition.py \
  --base 1000 --dim 32 --min-limbs 1 --max-limbs 4 \
  --local-pretrain-steps 5000 --steps 2000 \
  --out artifacts/latent_addition.json
```

The resulting artifact reports in-distribution integer/limb/carry accuracy and
the 1--4 limb training versus 1, 2, 4, 8, and 16 limb test matrix. This is a
learnability experiment, not an exact arithmetic claim: the exact conventional
limb implementation in `src/arithmetic/reference/` remains the oracle.

`--local-pretrain-steps` trains only the shared `(limb_a, limb_b, carry_in)`
transition before recurrent composition. It does not expose complete integers or
create position-specific weights.

For an exact local-rule diagnostic at a small base, add `--local-exhaustive`.
For example, base 32 contains exactly `2 × 32² = 2,048` local transitions per
pretraining step. Do not use this mode at base 1000 without batching it first.
The optional `--hard-case-repeat` reweights the maximal-sum boundary during an
exhaustive fine-tuning run; it does not change evaluation coverage.

Subtraction scaffolding is now in `models/subtraction.py`: it uses the same
shared local-transition principle, with an explicit borrow state. Run its first
base-10 experiment with `./.venv/bin/python train_subtraction.py`.

The first schoolbook multiplication experiment is available with
`./.venv/bin/python train_multiplication.py`. It reports the learned local
product-cell accuracy indirectly through final exact products at 1, 2, 4, and
8 limbs.

To evaluate the fully learned composition—learned local products plus the
learned recurrent carry adder rather than explicit multiplication normalization—run:

```bash
./.venv/bin/python evaluate_compositional_multiplication.py
```
