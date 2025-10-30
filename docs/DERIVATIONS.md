## Mathematical Derivations for V-SQLM

This document provides mathematical derivations and theoretical justification for the key components of V-SQLM.

---

## 1. Wilson One-Sided Confidence Bound

### Problem Setup

Given *n* samples with observed proportion *p*, compute a lower confidence bound for the true proportion *θ*.

### Derivation

The Wilson score interval is based on inverting the normal approximation test for proportions.

For a binomial proportion, the test statistic is:
```
Z = (p - θ) / √(θ(1-θ)/n)
```

For a one-sided confidence interval at level (1-α), we want:
```
P(Z < z_α) = 1 - α
```

Rearranging:
```
(p - θ) / √(θ(1-θ)/n) < z_α
```

Squaring both sides:
```
(p - θ)² < z_α² · θ(1-θ)/n
```

Expanding:
```
n(p - θ)² < z_α² · θ(1-θ)
np² - 2npθ + nθ² < z_α²θ - z_α²θ²
```

Rearranging into quadratic form:
```
(n + z_α²)θ² - (2np + z_α²)θ + np² < 0
```

Solving the quadratic equation `aθ² + bθ + c = 0` where:
- `a = n + z_α²`
- `b = -(2np + z_α²)`
- `c = np²`

The lower root (Wilson lower bound) is:
```
θ_lower = (-b - √(b² - 4ac)) / (2a)
        = (2np + z_α² - √((2np + z_α²)² - 4(n + z_α²)np²)) / (2(n + z_α²))
```

Simplifying:
```
θ_lower = (p + z²/(2n) - z√(p(1-p)/n + z²/(4n²))) / (1 + z²/n)
```

where `z = z_α`.

### Properties

1. **Convergence**: As n → ∞, θ_lower → p - z√(p(1-p)/n) (normal approximation)
2. **Monotonicity**: θ_lower increases with n (tighter bound with more data)
3. **Symmetry**: θ_upper for p equals θ_lower for (1-p)

### Example

For p = 0.7, n = 10, z = 1.96 (95% one-sided):
```
θ_lower = (0.7 + 1.96²/(2·10) - 1.96√(0.7·0.3/10 + 1.96²/(4·100))) / (1 + 1.96²/10)
        ≈ 0.465
```

With 95% confidence, the true proportion is at least 46.5%.

---

## 2. Beta-Binomial Majority Probability

### Problem Setup

Under a Beta-Binomial model, the success probability *p ~ Beta(α, β)*, and given *p*, the number of successes in *K* trials is *Binomial(K, p)*.

Compute: P(majority correct) = P(≥ ⌈K/2⌉ successes)

### Derivation

The Beta-Binomial distribution gives:
```
P(X = k) = C(K, k) · B(k + α, K - k + β) / B(α, β)
```

where B is the beta function:
```
B(a, b) = Γ(a)Γ(b) / Γ(a + b)
```

The probability of majority success is:
```
P_majority = Σ_{k=⌈K/2⌉}^K P(X = k)
           = Σ_{k=⌈K/2⌉}^K C(K, k) · B(k + α, K - k + β) / B(α, β)
```

In log space (for numerical stability):
```
log P(X = k) = log C(K, k) + log B(k + α, K - k + β) - log B(α, β)
             = [log Γ(K+1) - log Γ(k+1) - log Γ(K-k+1)]
               + [log Γ(k+α) + log Γ(K-k+β) - log Γ(K+α+β)]
               - [log Γ(α) + log Γ(β) - log Γ(α+β)]
```

### Interpretation

- **α/(α+β)**: Expected success probability
- **1/(α+β+1)**: Intra-class correlation ρ (sample dependence)
- **As ρ → 0**: Beta-Binomial → Binomial (independent samples)
- **As ρ → 1**: Samples become perfectly correlated

### Example

For α=8, β=2, K=5:
- Expected accuracy: 8/10 = 0.8
- ρ = 1/11 ≈ 0.09 (low dependence)
- P(majority correct) ≈ 0.99

For α=5, β=5, K=5:
- Expected accuracy: 0.5
- ρ = 1/11 ≈ 0.09
- P(majority correct) = 0.5 (symmetric case)

---

## 3. Peaked Difficulty Curve

### Motivation

To train a proposer to generate informative questions, we reward problems that are neither too easy nor too hard.

### Definition

Given solver success probability p*, the difficulty is:
```
d(p*) = 4p*(1 - p*)
```

### Properties

1. **Maximum at p* = 0.5**: d(0.5) = 1.0
2. **Zero at extremes**: d(0) = d(1) = 0
3. **Symmetry**: d(p) = d(1-p)
4. **Concave**: d''(p*) = -8 < 0 everywhere

### Derivation

This is the variance of a Bernoulli(p*) random variable, scaled by 4 to normalize:
```
Var(Bernoulli(p*)) = p*(1 - p*)
```

The factor 4 ensures maximum value 1:
```
max_{p*} 4p*(1 - p*) = 4 · 0.5 · 0.5 = 1
```

### Information-Theoretic Connection

The Shannon entropy of Bernoulli(p*) is:
```
H(p*) = -p* log p* - (1-p*) log(1-p*)
```

which is maximized at p* = 0.5 with H(0.5) = log 2.

The peaked difficulty approximates entropy shape while being simpler (quadratic vs transcendental).

---

## 4. Proposer Reward Function

### Full Reward

```
R = α · d(p*) + β · H(votes) + γ · 𝟙(incorrect) + δ · 𝟙(ill-posed)
```

Where:
- `d(p*)`: peaked difficulty (4p*(1-p*))
- `H(votes)`: vote entropy (Shannon entropy of vote distribution)
- `𝟙(incorrect)`: indicator for solver failure
- `𝟙(ill-posed)`: indicator for ambiguous/unsolvable problem

### Hyperparameters

Default values:
- `α = 1.0`: difficulty weight (positive incentive)
- `β = 0.5`: entropy weight (positive incentive for disagreement)
- `γ = -2.0`: incorrectness penalty
- `δ = -5.0`: ill-posed penalty (strong discouragement)

### Rationale

1. **Difficulty term (α·d(p*))**: Rewards problems at optimal difficulty where learning signal is strongest

2. **Entropy term (β·H)**: Rewards problems that elicit disagreement, indicating the problem tests model capabilities

3. **Incorrectness penalty (γ)**: Mild penalty for unsolved problems, encouraging solvable questions

4. **Ill-posed penalty (δ)**: Strong penalty for ambiguous or unsolvable problems, ensuring proposer learns to generate well-defined tasks

### Equilibrium

At equilibrium, the proposer generates problems with:
- p* ≈ 0.5 (optimal difficulty)
- High entropy (near-tie votes)
- Well-defined (not ill-posed)
- Eventually solvable (correct)

---

## 5. Calibration-Weighted Voting

### Empirical Bayes Approach

Track reliability statistics for each variant/model:
```
successes_i, trials_i
```

Under Beta(α_0, β_0) prior, the posterior is:
```
θ_i | data ~ Beta(α_0 + successes_i, β_0 + trials_i - successes_i)
```

The posterior mean (Bayes estimator) is:
```
w_i = E[θ_i | data] = (α_0 + successes_i) / (α_0 + β_0 + trials_i)
```

### Weighted Vote

Given equivalence classes with members, assign weight w_i to sample i:
```
W_c = Σ_{i ∈ members_c} w_i
```

Select class with maximum total weight:
```
c* = argmax_c W_c
```

### Prior Choice

Default: α_0 = β_0 = 1 (uniform prior)
- Regularizes toward 0.5 with limited data
- Posterior mean: (successes + 1) / (trials + 2)

Alternative: Jeffrey's prior α_0 = β_0 = 0.5
- Non-informative prior
- Posterior mean: (successes + 0.5) / (trials + 1)

---

## 6. Entropy Gating for Debate

### Vote Entropy

For equivalence classes with vote counts {n_1, n_2, ..., n_k}:
```
H = -Σ_i (n_i / N) log(n_i / N)
```

where N = Σ n_i.

### Normalized Entropy

To compare across different numbers of classes:
```
H_norm = H / log(k)
```

Range: [0, 1], where:
- 0: unanimous (one class has all votes)
- 1: uniform distribution (maximum disagreement)

### Gating Criterion

Trigger debate if:
```
H > τ  AND  no answer verified
```

Default τ = 0.4 (chosen empirically).

### Rationale

- **High entropy**: Indicates genuine disagreement where debate may help
- **Low entropy**: Strong consensus suggests debate unnecessary
- **Verified answer**: Debate unnecessary if answer already confirmed

### Example

4 samples, 2 classes:
- Case A: [3, 1] → H = -0.75 log 0.75 - 0.25 log 0.25 ≈ 0.56 → trigger
- Case B: [4, 0] → H = 0 → don't trigger
- Case C: [2, 2] → H = log 2 ≈ 0.69 → trigger

---

## 7. Expected Token Usage

### Fixed-K Self-Consistency

Expected tokens per task:
```
E[tokens] = K · (T_prompt + T_gen)
```

where T_prompt and T_gen are tokens per sample.

### V-SQLM with Adaptive Stopping

Expected tokens:
```
E[tokens] = E[K_stop] · (T_prompt + T_gen)
```

where K_stop is the stopping time.

### Expected Stopping Time

Under i.i.d. samples with agreement probability p:

**Wilson stopping at t**:
```
P(stop at t) ≈ P(Wilson_lower(X_t/t, t) > 0.5)
```

where X_t ~ Binomial(t, p).

For large p (high agreement):
```
E[K_stop] ≈ K_min + O(1)
```

For p ≈ 0.5 (low agreement):
```
E[K_stop] ≈ K_max
```

### Token Multiplier

Relative token usage vs fixed-K:
```
multiplier = E[K_stop] / K_fixed
```

Empirically:
- V-SQLM: 0.6-0.7× vs fixed-K=10
- V-SQLM+Debate: 0.7-0.8× vs fixed-K=10

---

## References

1. Wilson, E.B. (1927). "Probable inference, the law of succession, and statistical inference". JASA.

2. Skellam, J.G. (1948). "A probability distribution derived from the binomial distribution by regarding the probability of success as variable between the sets of trials". JRSS Series B.

3. Shannon, C.E. (1948). "A mathematical theory of communication". Bell System Technical Journal.

4. Gelman, A. et al. (2013). "Bayesian Data Analysis", 3rd edition. CRC Press.
