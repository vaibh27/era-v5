# Homomorphic Arithmetic Embeddings in Latent Space

## Detailed Research Report

**Status:** Theoretical formulation and experimental design

**Date:** 13 August 2026

---

## 1. Executive Summary

This report develops the research hypothesis **"Homomorphic Arithmetic Embeddings in Latent Space"** into a mathematically precise and experimentally testable framework.

The original idea asks whether a neural embedding function can be engineered so that arithmetic operations become intrinsic operations in latent space:

\[
E(x) \oplus E(y) = E(x+y)
\]

and

\[
E(x) \otimes E(y) = E(xy).
\]

The analysis establishes an important boundary.

The strongest unrestricted version of the hypothesis is **not achievable**: a finite-dimensional continuous latent representation cannot simply be turned into an unrestricted copy of real/integer arithmetic while preserving all arithmetic structure through a single simple latent operation. Continuous representations can exactly realize individual algebraic structures, and finite algebraic domains can realize both addition and multiplication simultaneously. The uploaded research identifies this boundary through rigidity results, the distinction between real and integer arithmetic, capacity limitations, and undecidability results.

The viable version of the hypothesis is substantially stronger scientifically because it is precise:

> **A dedicated arithmetic subspace of a neural embedding can be engineered or learned so that, over a finite/bounded arithmetic domain, fixed latent operations realize addition and multiplication exactly, while a separate semantic subspace retains linguistic information.**

The constructive mathematical route developed here is based on finite algebraic structures and the Chinese Remainder Theorem (CRT)/Residue Number Systems (RNS). The resulting arithmetic representation is an exact ring representation over a bounded domain. The research then proposes to bridge this exact mathematical construction into a continuous neural embedding and finally concatenate it with a semantic representation.

The report therefore separates five stages:

1. **Formal theorem** — existence of an exact arithmetic subspace over a finite/bounded domain.
2. **Lemmas and proofs** — injectivity, addition preservation, multiplication preservation, capacity, and continuous realization.
3. **Architecture** — semantic and arithmetic subspaces with constrained arithmetic operators.
4. **Experiments** — tests for latent homomorphism, extrapolation, ring laws, capacity, and semantic preservation.
5. **Open problem** — whether gradient-based learning can discover and maintain the structured arithmetic representation efficiently.

---

## 2. Original Research Hypothesis

The original hypothesis proposes continuous vector representations that preserve strict algebraic structure.

For numeric tokens \(x\) and \(y\), the desired properties are:

\[
E(x) \oplus E(y)=E(x+y)
\]

and

\[
E(x) \otimes E(y)=E(x\times y).
\]

The motivating examples are:

\[
E(9)\oplus E(9)=E(18)
\]

and

\[
E(9)\otimes E(9)=E(81).
\]

The original proposal also introduced a partitioned embedding:

\[
E_{total}(x)=
[E_{semantic}(x)\parallel E_{math}(x)].
\]

The semantic subspace is intended for ordinary linguistic features, while the arithmetic subspace is explicitly constrained to preserve mathematical properties and operations.

The research question was therefore not merely whether a neural model can *predict* arithmetic answers, but whether the **latent representation itself can contain an algebraic structure**.

---

## 3. Research Basis and Scope

The uploaded research review provides the principal theoretical basis for the work summarized here. It identifies several relevant results and prior-art directions:

- exact additive embeddings;
- multiplicative/logarithmic representations;
- rigidity/no-go results for simultaneous real-field structure;
- finite-field constructions;
- CRT/RNS arithmetic;
- Fourier/clock representations;
- neural arithmetic units;
- vector symbolic/hyperdimensional representations;
- capacity limitations;
- fixed-depth computational limitations;
- undecidability boundaries;
- existing numeric embedding methods including xVal, FoNE, Abacus and related work.

Source basis: uploaded research review, especially its sections on the core findings, homomorphic embeddings, CRT/RNS, fundamental limits, neural numeric embeddings, and recommendations.

---

# Part I — Formal Mathematical Formulation

## 4. Full Embedding Space and Arithmetic Subspace

Let the complete neural embedding be

\[
E:\mathcal X\rightarrow\mathbb R^d.
\]

The latent space decomposes into two subspaces:

\[
\mathbb R^d=V_{semantic}\oplus V_{math}.
\]

Accordingly,

\[
E(x)=
[E_{semantic}(x)\parallel E_{math}(x)].
\]

The roles are intentionally distinct:

### Semantic subspace

\[
E_{semantic}(x)\in V_{semantic}
\]

stores linguistic/distributional information.

### Arithmetic subspace

\[
E_{math}(x)\in V_{math}
\]

is intended to realize a specific algebraic structure.

Define fixed operations

\[
\oplus:V_{math}\times V_{math}\rightarrow V_{math}
\]

and

\[
\otimes:V_{math}\times V_{math}\rightarrow V_{math}.
\]

The desired homomorphism conditions are

\[
E_{math}(x+y)=E_{math}(x)\oplus E_{math}(y)
\]

and

\[
E_{math}(xy)=E_{math}(x)\otimes E_{math}(y).
\]

The central point is that these identities concern the **arithmetic subspace**, not the entire semantic embedding.

---

# Part II — Formal Theorem

## 5. Exact Arithmetic Subspace Embedding Theorem

### Theorem

Let \(m_1,\ldots,m_k\) be pairwise coprime positive integers and define

\[
M=\prod_{i=1}^{k}m_i.
\]

Let

\[
S=\mathbb Z_M.
\]

Then there exists an injective arithmetic embedding

\[
E_{math}:\mathbb Z_M
\rightarrow
\mathbb Z_{m_1}\times\cdots\times\mathbb Z_{m_k}
\]

such that fixed operations \(\oplus\) and \(\otimes\) satisfy

\[
E_{math}(x+y)=E_{math}(x)\oplus E_{math}(y)
\]

and

\[
E_{math}(xy)=E_{math}(x)\otimes E_{math}(y)
\]

for every \(x,y\in\mathbb Z_M\), where operations on the right are performed component-wise modulo each corresponding modulus.

Therefore, a bounded arithmetic domain admits a faithful latent representation in which both addition and multiplication are realized exactly by fixed operations on the representation.

### Interpretation

This is the mathematically safe version of the original hypothesis.

It does **not** assert that arbitrary unbounded \(\mathbb Z\) or \(\mathbb R\) can be represented as one finite continuous arithmetic space with no limitations.

It asserts a constructive existence result for a finite/bounded algebraic domain.

---

# Part III — Lemmas and Proofs

## 6. Lemma 1 — Injectivity of the CRT Arithmetic Embedding

### Definition

Define

\[
E_{math}(x)=
(x\bmod m_1,\ldots,x\bmod m_k).
\]

### Lemma

The mapping \(E_{math}\) is injective on \(\mathbb Z_M\).

### Proof

Assume

\[
E_{math}(a)=E_{math}(b).
\]

Then, for every \(i\),

\[
a\bmod m_i=b\bmod m_i.
\]

Therefore,

\[
m_i\mid(a-b)
\]

for every \(i\).

Because the moduli are pairwise coprime,

\[
\prod_{i=1}^{k}m_i\mid(a-b).
\]

But

\[
M=\prod_{i=1}^{k}m_i,
\]

so

\[
M\mid(a-b).
\]

Thus

\[
a\equiv b\pmod M.
\]

Since the domain is \(\mathbb Z_M\),

\[
a=b.
\]

Therefore,

\[
E_{math}(a)=E_{math}(b)\Rightarrow a=b,
\]

so the embedding is injective.

\[
\boxed{\square}
\]

### Meaning

No information is lost within the chosen bounded arithmetic domain. Different arithmetic values map to different latent states.

---

## 7. Lemma 2 — Additive Homomorphism

Define the latent addition operation

\[
(u_1,\ldots,u_k)\oplus(v_1,\ldots,v_k)
=
((u_1+v_1)\bmod m_1,\ldots,(u_k+v_k)\bmod m_k).
\]

### Lemma

For all \(x,y\in\mathbb Z_M\),

\[
E_{math}(x+y)=E_{math}(x)\oplus E_{math}(y).
\]

### Proof

By definition,

\[
E_{math}(x+y)
=
((x+y)\bmod m_1,\ldots,(x+y)\bmod m_k).
\]

For each modulus,

\[
(x+y)\bmod m_i
=
((x\bmod m_i)+(y\bmod m_i))\bmod m_i.
\]

Therefore the complete vector is exactly the component-wise modular sum:

\[
E_{math}(x+y)=E_{math}(x)\oplus E_{math}(y).
\]

\[
\boxed{\square}
\]

### Example

Use

\[
(m_1,m_2,m_3)=(5,7,11).
\]

Then

\[
E(9)=(4,2,9).
\]

Hence

\[
E(9)\oplus E(9)
=(4+4\bmod5,2+2\bmod7,9+9\bmod11)
=(3,4,7).
\]

And

\[
E(18)=(3,4,7).
\]

Therefore,

\[
\boxed{E(9)\oplus E(9)=E(18)}.
\]

---

## 8. Lemma 3 — Multiplicative Homomorphism

Define the latent multiplication operation

\[
(u_1,\ldots,u_k)\otimes(v_1,\ldots,v_k)
=
((u_1v_1)\bmod m_1,\ldots,(u_kv_k)\bmod m_k).
\]

### Lemma

For all \(x,y\in\mathbb Z_M\),

\[
E_{math}(xy)=E_{math}(x)\otimes E_{math}(y).
\]

### Proof

By definition,

\[
E_{math}(xy)
=((xy)\bmod m_1,\ldots,(xy)\bmod m_k).
\]

For every modulus,

\[
xy\bmod m_i
=
((x\bmod m_i)(y\bmod m_i))\bmod m_i.
\]

Thus,

\[
E_{math}(xy)=E_{math}(x)\otimes E_{math}(y).
\]

\[
\boxed{\square}
\]

### Example

Using the same moduli,

\[
E(9)=(4,2,9).
\]

Then

\[
E(9)\otimes E(9)
=(4^2\bmod5,2^2\bmod7,9^2\bmod11)
=(1,4,4).
\]

Since

\[
E(81)=(1,4,4),
\]

the result is

\[
\boxed{E(9)\otimes E(9)=E(81)}.
\]

---

## 9. Corollary — Faithful Ring Representation

Lemmas 1–3 imply that

\[
E_{math}:\mathbb Z_M
\hookrightarrow
\prod_{i=1}^{k}\mathbb Z_{m_i}
\]

is injective and preserves both addition and multiplication.

Thus it is a faithful ring representation of the bounded/modular arithmetic structure.

The core existence statement is therefore proven:

\[
\boxed{
E_{math}(x+y)=E_{math}(x)\oplus E_{math}(y),
\quad
E_{math}(xy)=E_{math}(x)\otimes E_{math}(y).
}
\]

---

## 10. Lemma 4 — Capacity of the Arithmetic Subspace

The representation has exactly

\[
M=\prod_{i=1}^{k}m_i
\]

possible states.

Each residue coordinate has \(m_i\) possible values, so the Cartesian product contains

\[
\prod_i m_i=M
\]

states.

Injectivity ensures that each state corresponds to a unique element of \(\mathbb Z_M\).

Therefore the arithmetic representation can encode exactly \(M\) distinct arithmetic values.

\[
\boxed{\square}
\]

---

## 11. Lemma 5 — Exponential Capacity with the Number of Residue Channels

If the moduli are of roughly comparable magnitude \(q\), then

\[
M=\prod_{i=1}^{k}m_i\approx q^k.
\]

Hence,

\[
\log M\approx k\log q.
\]

Therefore the number of channels needed to represent a range of size \(M\) scales logarithmically with the range, while the representable range grows exponentially with the number of residue channels.

This is the fundamental capacity advantage of CRT/RNS representations identified in the research review.

---

## 12. Lemma 6 — Continuous Vector Realization

The discrete residue representation can be mapped into continuous real vectors using circular/Fourier coordinates.

For modulus \(m_i\), define

\[
\phi_i(r)=
\left(
\cos\frac{2\pi r}{m_i},
\sin\frac{2\pi r}{m_i}
\right).
\]

Then define

\[
E_{cont}(x)=
[\phi_1(x\bmod m_1)\parallel\cdots\parallel\phi_k(x\bmod m_k)].
\]

Thus,

\[
E_{cont}(x)\in\mathbb R^{2k}.
\]

Equivalently, in complex form,

\[
z_i(x)=e^{2\pi i(x\bmod m_i)/m_i}.
\]

Then

\[
z_i(x)z_i(y)
=z_i(x+y).
\]

So modular addition can be represented as fixed phase multiplication/rotation.

### Important qualification

This construction directly provides a continuous representation of the residue channels and makes modular addition geometrically simple. It does **not by itself prove** that arbitrary multiplication can always be represented by a comparably simple low-dimensional Euclidean operator. Exact multiplication is guaranteed in the residue algebra; obtaining an efficient continuous/tensor realization is a separate construction problem and experimental target.

---

## 13. Lemma 7 — Bilinear/Tensor Realization of Multiplication

A direct continuous-vector construction can be obtained by representing residue states as basis vectors.

For a modulus \(m\), use

\[
h_m(r)\in\{0,1\}^{m}.
\]

Define a bilinear map

\[
B_m:\mathbb R^m\times\mathbb R^m\rightarrow\mathbb R^m
\]

on basis states by

\[
B_m(e_a,e_b)=e_{ab\bmod m}.
\]

Extend bilinearly.

Then

\[
B_m(h_m(x),h_m(y))
=h_m(xy\bmod m).
\]

Stacking all residue channels gives a fixed tensor operator \(B\) satisfying

\[
B(E(x),E(y))=E(xy).
\]

In coordinates, a rank-3 tensor \(W\) can implement

\[
[T(u,v)]_k
=
\sum_{i,j}W_{kij}u_i v_j.
\]

This demonstrates the precise form of the original tensor-operation hypothesis.

### Qualification

The one-hot construction is exact but may be dimensionally inefficient. The research problem is to determine whether more compact structured continuous representations can preserve both operations while remaining trainable and robust.

---

# Part IV — Theoretical Limits

## 14. Why the Entire Continuous Embedding Space Cannot Simply Become Arithmetic

The original unrestricted formulation asks for a continuous latent system that behaves as arithmetic over unbounded \(\mathbb Z\) or \(\mathbb R\).

The research review identifies a rigidity/no-go boundary:

- continuous additive representations of \(\mathbb R\) are highly constrained by Cauchy-type rigidity;
- simultaneous preservation of both addition and multiplication severely restricts continuous ring endomorphisms;
- the multiplicative structure of \(\mathbb R\) has an essential zero/non-invertibility structure that differs from the additive group;
- finite-dimensional continuous representations cannot be assumed to recreate all arithmetic structure through one arbitrary simple operation.

The consequence for this project is:

> **Do not claim that the full continuous embedding space can become unrestricted arithmetic.**

Instead, engineer a dedicated arithmetic subspace with a specific finite/bounded algebraic structure.

---

## 15. Multiplicative and Additive Channels Cannot Simply Be Naively Concatenated

A tempting representation is

\[
E(x)=[x,\log x].
\]

The linear channel naturally handles addition, while the log channel naturally handles multiplication.

But after addition,

\[
[x,\log x]+[y,\log y]
=[x+y,\log x+\log y]
=[x+y,\log(xy)].
\]

The correct representation of \(x+y\) would be

\[
[x+y,\log(x+y)].
\]

Therefore the channels are not automatically synchronized.

The required transformation involves a nonlinearity of the log-sum-exp type.

This is a fundamental channel-coherence issue and is not solved merely by concatenating two mathematically useful encodings.

---

## 16. Capacity and Unbounded Arithmetic

A finite-dimensional representation with finite precision has finite effective capacity.

Therefore exact representation of an unbounded set of distinct integers cannot be achieved with a fixed finite number of reliably distinguishable states unless a different resource—such as precision, recurrence, time, or external computation—also scales.

CRT/RNS changes the tradeoff by storing residues. It gives an exponentially growing finite range in the number of channels, but the domain remains bounded by the product of the moduli.

Therefore:

\[
\boxed{
\text{bounded exact arithmetic is feasible; unrestricted exact arithmetic needs additional resources.}
}
\]

---

## 17. Computational Depth Limitation

The uploaded research also identifies a fixed-depth computational boundary for unbounded arithmetic. Multiplication and addition are computationally tractable, but carrying, serial composition, and unbounded extrapolation create architectural demands.

Therefore, an exact architecture beyond a bounded residue range may require:

- recurrence;
- looped computation;
- explicit carry handling;
- increased depth;
- chain-of-thought or iterative computation;
- or an external symbolic arithmetic mechanism.

This provides a second theoretical reason not to claim unlimited arithmetic from a fixed-width, fixed-depth embedding.

---

## 18. Undecidability Boundary

The research review distinguishes real arithmetic from integer arithmetic.

The first-order theory of real-closed fields is decidable, whereas exact integer arithmetic is connected to undecidable problems such as Hilbert's tenth problem and the undecidability/incompleteness results associated with arithmetic.

The relevant research boundary is:

> A finite-dimensional continuous representation may capture useful algebraic structure, but it cannot turn all arithmetic truth into a finite, decidable latent computation.

This does **not** imply that mathematical syntax cannot be encoded. Gödel-style numbering demonstrates that countable syntax can be represented exactly. The limitation concerns full semantic truth and decision procedures, not mere encoding.

---

# Part V — Refined Research Hypothesis

## 19. Revised Hypothesis

The original hypothesis should be narrowed to the following statement:

> **A neural embedding can be decomposed into a semantic subspace and a dedicated arithmetic subspace such that the arithmetic subspace realizes an exact finite algebraic representation, with fixed latent operations corresponding homomorphically to arithmetic operations over a bounded domain, while the semantic subspace remains independently usable for linguistic representation.**

This is the hypothesis that the architecture and experiments can directly test.

---

# Part VI — Proposed Architecture

## 20. Overall Architecture

The complete embedding is

\[
E_{total}(x)
=
[E_{semantic}(x)\parallel E_{math}(x)].
\]

Conceptually:

```text
                         Input token / number
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
          Semantic Encoder             Arithmetic Encoder
                  |                           |
                  v                           v
          E_semantic(x)              E_math(x)
                                              |
                                   +----------+----------+
                                   |                     |
                                   v                     v
                              Arithmetic +         Arithmetic x
                                   |                     |
                                   +----------+----------+
                                              |
                                              v
                                  Structured math state
                                              |
                         +--------------------+
                         |
                         v
                [E_semantic || E_math]
```

The arithmetic operators act only on the arithmetic subspace.

---

## 21. Semantic Subspace

Let

\[
E_{semantic}(x)\in\mathbb R^{d_s}.
\]

This component is learned using ordinary semantic objectives such as similarity, retrieval, classification, contextual prediction, or language-model losses.

An illustrative initial size is

\[
d_s=32,
\]

but the exact value is experimental rather than theoretically required.

---

## 22. Arithmetic Subspace

Let

\[
E_{math}(x)\in\mathbb R^{d_m}.
\]

This subspace has explicit structural constraints.

Two operators are defined:

\[
A_+(u,v)
\]

and

\[
A_\times(u,v).
\]

The desired relations are

\[
A_+(E_{math}(x),E_{math}(y))
\approx E_{math}(x+y)
\]

and

\[
A_\times(E_{math}(x),E_{math}(y))
\approx E_{math}(xy).
\]

For an exact algebraic construction, the approximation becomes equality.

---

## 23. Candidate Addition Operator

Possible implementations include:

1. component-wise modular addition;
2. phase rotation in circular/Fourier channels;
3. another fixed representation of the finite group structure.

The first prototype should use a mathematically transparent operator rather than a general neural network.

---

## 24. Candidate Multiplication Operator

A tensor/bilinear operator is a natural candidate:

\[
[A_\times(u,v)]_k
=
\sum_{i,j}W_{kij}u_i v_j.
\]

The tensor \(W\) can be fixed from a known algebra or learned under algebraic constraints.

This makes the original idea of a tensor product operation explicit.

---

## 25. Arithmetic Training Objective

Given operands \(x,y\), calculate

\[
\hat e_+=A_+(E_m(x),E_m(y))
\]

and

\[
\hat e_\times=A_\times(E_m(x),E_m(y)).
\]

Use

\[
\mathcal L_+
=\|\hat e_+-E_m(x+y)\|^2
\]

and

\[
\mathcal L_\times
=\|\hat e_\times-E_m(xy)\|^2.
\]

Then

\[
\mathcal L_{arith}
=\mathcal L_+ + \mathcal L_\times.
\]

---

## 26. Separation / Injectivity Objective

The embedding must avoid collapsing distinct values.

A margin-based separation loss can be used:

\[
\mathcal L_{sep}
=
\sum_{x\neq y}
\max\left(0,\delta-\|E_m(x)-E_m(y)\|\right)^2.
\]

The total arithmetic objective can therefore be

\[
\mathcal L
=
\mathcal L_{arith}
+\lambda_{sep}\mathcal L_{sep}.
\]

---

## 27. Semantic + Arithmetic Joint Objective

When the two subspaces are combined,

\[
\mathcal L_{total}
=
\mathcal L_{semantic}
+\lambda_a\mathcal L_{arith}
+\lambda_s\mathcal L_{sep}.
\]

The semantic loss tests language performance; the arithmetic losses test algebraic structure.

This allows the research to ask whether the arithmetic constraints can be localized to the math subspace without materially degrading semantic behavior.

---

## 28. Decoder

Define a decoder

\[
D:V_{math}\rightarrow S.
\]

The desired property is

\[
D(E_m(x))=x.
\]

For operations,

\[
D(A_+(E_m(x),E_m(y)))=x+y
\]

and

\[
D(A_\times(E_m(x),E_m(y)))=xy
\]

within the supported domain.

Two separate evaluation levels are required:

1. **latent-level correctness**;
2. **decoded-answer correctness**.

This distinction prevents a decoder from hiding errors in the latent arithmetic structure.

---

# Part VII — Experimental Program

## 29. Experiment 1 — Exact Mathematical Sanity Check

Choose pairwise-coprime moduli, for example:

\[
(5,7,11,13).
\]

Then

\[
M=5005.
\]

Represent

\[
0,1,\ldots,5004.
\]

Exhaustively verify:

\[
E((x+y)\bmod M)=E(x)\oplus E(y)
\]

and

\[
E((xy)\bmod M)=E(x)\otimes E(y).
\]

Expected outcome:

\[
\boxed{100\%\text{ exactness}.}
\]

This verifies the implementation of the theorem before any learning is introduced.

---

## 30. Experiment 2 — Learn the Arithmetic Embedding

Remove the explicit arithmetic lookup and learn

\[
E_\theta(x)\in\mathbb R^{d_m}.
\]

Train the embedding using arithmetic consistency losses.

The initial model should contain no transformer and no natural-language component.

The objective is to determine whether gradient-based optimization can learn a structured representation rather than simply memorize number-to-answer examples.

---

## 31. Experiment 3 — Held-Out Operand Combinations

Hold out operand pairs rather than entire numbers.

For example, the model may observe individual numbers during training but not a particular pair.

Test whether

\[
A_+(E(12),E(14))
\]

produces

\[
E(26)
\]

without having explicitly observed that particular pair.

Similarly test multiplication.

This isolates **compositional generalization**.

---

## 32. Experiment 4 — Range Extrapolation

Train on

\[
0\le x,y\le100
\]

and test on larger values, subject to the modular capacity boundary.

Examples include:

\[
347+528
\]

and

\[
37\times241.
\]

The evaluation must distinguish values within the representational capacity from values that exceed the CRT modulus.

---

## 33. Experiment 5 — Latent Homomorphism Error

Define

\[
\epsilon_+(x,y)
=
\|A_+(E(x),E(y))-E(x+y)\|
\]

and

\[
\epsilon_\times(x,y)
=
\|A_\times(E(x),E(y))-E(xy)\|.
\]

Report:

- mean error;
- median error;
- maximum error;
- percentile distribution;
- exact-match rate when appropriate.

This is the primary evidence for or against the latent homomorphism hypothesis.

---

## 34. Experiment 6 — Associativity

Test whether the learned addition satisfies

\[
(E(x)\oplus E(y))\oplus E(z)
=
E(x)\oplus(E(y)\oplus E(z)).
\]

Likewise for multiplication.

Define the corresponding latent associativity errors.

Success on unseen triples is stronger evidence of an algebraic structure than pairwise accuracy alone.

---

## 35. Experiment 7 — Distributivity

Test the ring law

\[
x(y+z)=xy+xz.
\]

At the representation level, evaluate

\[
E(x)\otimes(E(y)\oplus E(z))
\]

against

\[
(E(x)\otimes E(y))\oplus(E(x)\otimes E(z)).
\]

Small error across unseen triples would support the claim that the system learned a coherent ring-like structure rather than two unrelated arithmetic mappings.

---

## 36. Experiment 8 — Semantic Isolation

Compare:

### Baseline

\[
E_{baseline}=E_{semantic}.
\]

### Structured model

\[
E_{structured}
=[E_{semantic}\parallel E_{math}].
\]

Evaluate semantic retrieval, similarity, classification, or language-model performance.

The target is:

\[
\Delta semantic\approx0
\]

while arithmetic performance improves substantially.

---

## 37. Experiment 9 — Ablation Study

Compare at minimum:

1. ordinary learned embedding;
2. arithmetic-only embedding;
3. concatenated semantic + arithmetic embedding;
4. concatenated embedding with no arithmetic structure;
5. fully structured arithmetic embedding.

The desired result is that the structured architecture provides arithmetic benefits without sacrificing the semantic task.

---

## 38. Experiment 10 — Dimension Scaling

Vary the arithmetic dimension:

\[
d_m=4,8,16,32,64,\ldots
\]

and estimate maximum reliably exact range.

Plot

\[
d_m\quad\text{vs}\quad M_{max}.
\]

This tests the theoretical expectation that structured residue representations can achieve rapidly increasing range as the number of independent channels increases.

---

## 39. Experiment 11 — Noise Robustness

Perturb the learned representation:

\[
\tilde E(x)=E(x)+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma^2 I).
\]

Measure arithmetic accuracy as a function of \(\sigma\).

This determines whether the structure is merely mathematically exact in ideal coordinates or robust enough to survive realistic neural representation noise.

---

## 40. Experiment 12 — Comparison With Existing Approaches

The research review identifies several relevant baselines and prior directions:

- xVal;
- FoNE;
- Abacus;
- NALU/NAC and related neural arithmetic units;
- Fourier/clock-based numeric representations;
- residue/hyperdimensional representations.

The evaluation should use the same arithmetic tasks and report both ordinary task accuracy and latent algebraic fidelity.

A model that predicts the right answer but does not satisfy the latent homomorphism equations should not be considered equivalent to the proposed structured embedding.

---

# Part VIII — Evaluation Criteria

## 41. What Would Count as Evidence for the Hypothesis?

A strong empirical result would satisfy several criteria simultaneously.

### Criterion 1 — Arithmetic exactness

For held-out arithmetic compositions,

\[
P(\hat z=z)\approx100\%.
\]

### Criterion 2 — Latent homomorphism


\[
\epsilon_+,\epsilon_\times\rightarrow0.
\]

### Criterion 3 — Algebraic laws

Associativity and distributivity hold approximately/exactly on unseen examples.

### Criterion 4 — Compositional generalization

The model succeeds on operand combinations not seen during training.

### Criterion 5 — Capacity scaling

Increasing arithmetic subspace size expands the exact range in a predictable way.

### Criterion 6 — Semantic preservation

The arithmetic constraints do not materially damage the semantic benchmark.

---

# Part IX — What the Research Has Actually Established

## 42. Established Mathematical Results

The work completed so far establishes the following conceptual sequence:

\[
\boxed{\text{Arithmetic algebra}}
\rightarrow
\boxed{\text{finite residue representation}}
\rightarrow
\boxed{\text{exact + and ×}}
\rightarrow
\boxed{\text{continuous-vector realization}}
\rightarrow
\boxed{\text{neural learning problem}}.
\]

More specifically:

1. Exact additive embeddings are straightforward.
2. Exact multiplicative embeddings can be constructed separately, e.g. via logarithmic representations on positive reals.
3. Naively concatenating additive and multiplicative channels creates a channel-coherence problem.
4. Finite algebraic structures can realize both operations simultaneously.
5. CRT/RNS provides an exact bounded-integer construction with component-wise addition and multiplication.
6. The number of representable values grows as the product of the moduli.
7. Circular/Fourier encodings provide a continuous geometric realization of residue channels.
8. Tensor/bilinear operations can represent multiplication exactly in a suitable basis.
9. Unbounded arithmetic and full arithmetic truth require resources beyond the simple fixed-dimensional continuous model.
10. The semantic + arithmetic concatenation is therefore best treated as a structured architecture rather than a claim that the entire embedding space becomes arithmetic.

---

# Part X — Important Non-Claims

## 43. What Has Not Been Proven

The current work does **not** prove that:

- a standard LLM will spontaneously discover the desired arithmetic representation;
- a low-dimensional Euclidean embedding can implement arbitrary unbounded multiplication exactly;
- semantic and arithmetic subspaces will automatically remain orthogonal without explicit constraints;
- a learned tensor operator will converge to the theoretically correct algebra;
- the representation will generalize indefinitely beyond its finite capacity;
- the model will encode all arithmetic truth;
- the proposed architecture is already better than all existing numeric embedding methods.

These are empirical or future theoretical questions.

---

# Part XI — Proposed Research Roadmap

## 44. Stage 1 — Mathematical Verification

Implement and exhaustively verify the CRT/RNS construction.

Goal:

\[
100\%\text{ exact arithmetic within the finite domain}.
\]

---

## 45. Stage 2 — Neural Arithmetic Embedding

Replace the hand-defined embedding with a learnable embedding and test whether the algebraic structure can be learned.

Goal:

\[
A_+(E(x),E(y))\approx E(x+y)
\]

and

\[
A_\times(E(x),E(y))\approx E(xy).
\]

---

## 46. Stage 3 — Algebraic Stress Tests

Test associativity, distributivity, extrapolation, latent error, and noise robustness.

Goal: establish that the model learned an algebraic system rather than a lookup strategy.

---

## 47. Stage 4 — Semantic + Arithmetic Embedding

Introduce

\[
E_{total}=[E_{semantic}\parallel E_{math}].
\]

Goal: preserve semantic quality while gaining exact/near-exact arithmetic behavior.

---

## 48. Stage 5 — Natural-Language Integration

Map natural-language numeric tokens such as

"nine"

and

"9"

to a common arithmetic representation.

The eventual target is a model where linguistic processing and arithmetic processing can coexist in the same overall embedding while arithmetic operations remain localized to the dedicated subspace.

---

## 49. Stage 6 — Larger Models and Modern Numeric Embeddings

Only after the small controlled experiments succeed should the work be integrated into transformer architectures and compared directly to contemporary numeric embedding schemes identified in the research review.

---

# Part XII — Final Research Position

## 50. Final Conclusion

The original intuition was partially correct but too broad.

It is not defensible to claim:

> "The entire continuous embedding space can become arithmetic and exactly model unrestricted integer/real mathematics."

The defensible and much more interesting claim is:

> **A neural embedding can contain a dedicated, explicitly structured arithmetic subspace that faithfully represents a bounded algebraic domain and realizes arithmetic operations through fixed latent operators, while the remaining subspace continues to encode ordinary semantic information.**

The mathematical foundation is strongest in finite/bounded algebraic settings such as CRT/RNS representations and finite fields. These structures give genuine constructive existence proofs rather than heuristic arguments.

The central open problem is then shifted from pure existence to **learnability and integration**:

\[
\boxed{
\text{Can gradient-based neural training discover and maintain a useful algebraic subspace?}
}
\]

and, subsequently,

\[
\boxed{
\text{Can that subspace coexist with semantic representations without sacrificing either capability?}
}
\]

These are experimentally falsifiable questions and provide a clear path from theorem to implementation.

---

# Appendix A — Compact Mathematical Summary

### Arithmetic domain

\[
S=\mathbb Z_M,
\qquad
M=\prod_i m_i,
\qquad
\gcd(m_i,m_j)=1.
\]

### Arithmetic embedding

\[
E_m(x)=
(x\bmod m_1,\ldots,x\bmod m_k).
\]

### Latent addition

\[
(u\oplus v)_i=(u_i+v_i)\bmod m_i.
\]

### Latent multiplication

\[
(u\otimes v)_i=(u_iv_i)\bmod m_i.
\]

### Homomorphism equations

\[
E_m(x+y)=E_m(x)\oplus E_m(y)
\]

\[
E_m(xy)=E_m(x)\otimes E_m(y).
\]

### Injectivity

\[
E_m(x)=E_m(y)\Rightarrow x=y.
\]

### Capacity

\[
|S|=M=\prod_i m_i.
\]

### Full embedding

\[
E_{total}(x)
=[E_{semantic}(x)\parallel E_m(x)].
\]

### Neural objective

\[
\mathcal L_{total}
=
\mathcal L_{semantic}
+\lambda_a(\mathcal L_++\mathcal L_\times)
+\lambda_s\mathcal L_{sep}.
\]

---

# Appendix B — Key Examples

Using moduli

\[
(5,7,11):
\]

\[
E(9)=(4,2,9).
\]

### Addition

\[
E(9)\oplus E(9)
=(3,4,7)
=E(18).
\]

### Multiplication

\[
E(9)\otimes E(9)
=(1,4,4)
=E(81).
\]

These are exact within the chosen modular domain.

---

# Appendix C — Relationship to the Uploaded Research Review

The uploaded research review supports the following core points used in this report:

- exact additive embeddings and clock/Fourier representations;
- the rigidity/no-go argument for unrestricted continuous simultaneous + and × representations;
- finite-field existence proofs;
- CRT/RNS as the exact bounded arithmetic route;
- channel-coherence limitations of naïve log/linear dual representations;
- vector-symbolic and hyperdimensional arithmetic constructions;
- capacity and fixed-depth computational limits;
- the distinction between arithmetic structure and full mathematical truth;
- prior art including xVal, FoNE, Abacus and neural arithmetic units;
- the proposed open direction combining exact residue arithmetic with a semantic subspace.

The research review itself flags several very recent/pre-publication claims as requiring independent verification before being treated as load-bearing claims. Those caveats should be preserved in any formal paper submission.

---

# Appendix D — Core Scientific Question

After the theoretical groundwork, the most important experimental question is:

\[
\boxed{
\text{Can a small neural model learn an embedding }E_m\text{ and operators }A_+,A_\times
}
\]

such that, on unseen compositions,

\[
A_+(E_m(x),E_m(y))\approx E_m(x+y)
\]

and

\[
A_\times(E_m(x),E_m(y))\approx E_m(xy),
\]

while also satisfying higher-order algebraic laws such as associativity and distributivity?

If yes, the next question is whether this learned arithmetic subspace can be concatenated with semantic embeddings without degrading language performance.

That is the experimental core of the proposed research.
