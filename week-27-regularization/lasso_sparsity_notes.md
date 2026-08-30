# Why Lasso Creates Sparsity — Full Derivation

## Setup

For simple linear regression with Lasso penalty, we minimize:

```
Loss = Σ(Yi - (mXi + b))² + alpha·|m|
```

Let `S = Σ(Yi - Y_mean)(Xi - X_mean)` and `D = Σ(Xi - X_mean)²`.
Note: `S/D` is just the OLS slope (no regularization).

The `|m|` term is not differentiable at m=0, so we cannot take one derivative
for all m. We split into cases.

---

## Step 1: Split into three cases (this is a *subgradient* problem)

Because `|m|` behaves differently depending on the sign of m, we treat the
optimization as three separate **hypotheses**, each solved independently and
then checked for consistency:

**Case m > 0:** here `|m| = m`, so differentiating gives:

```
m = (S - alpha) / D      ... valid ONLY IF this comes out > 0
```

**Case m < 0:** here `|m| = -m`, so differentiating gives:

```
m = (S + alpha) / D      ... valid ONLY IF this comes out < 0
```

**Case m = 0:** no differentiation needed — just check if m=0 is consistent
(i.e. both above cases fail their own sign requirement).

This is the key idea: **each formula is a hypothesis, not a guaranteed
answer.** You solve assuming a sign, then check whether the result actually
has that sign. If it doesn't, that case is thrown out.

---

## Step 2: Why increasing alpha kills m (sparsity)

Assume S > 0 (positive correlation, so OLS slope would be positive).

- Small alpha: `m = (S - alpha)/D` is still positive → this is the valid case. m shrinks as alpha grows.
- At `alpha = S`: m becomes exactly 0.
- If alpha keeps increasing past S: `(S - alpha)/D` goes negative — but this
  formula was only valid *for the m>0 case*. A negative result violates its
  own assumption, so it's **rejected**, not used.
- Check the m<0 case instead: `m = (S + alpha)/D`. Since alpha is large and
  positive, this is even more positive — it also violates its own assumption
  (it needs to be negative). **Also rejected.**
- Since neither the positive-case nor negative-case formula is
  self-consistent, the only remaining answer is **m = 0**.

This is why sklearn never gives you a negative m in this scenario, even
though naively plugging into `(S - alpha)/D` "looks like" it should go
negative — that formula stops being valid the moment it crosses zero.

**General threshold condition** (covers both signs of S, since alpha ≥ 0 always):

```
m = 0   whenever   alpha ≥ |S|
```

This is exactly the *soft-thresholding* operation — the mathematical
mechanism that gives Lasso its sparsity (Ridge, by contrast, has no such
hard zero-point because it penalizes m², which is differentiable everywhere).

---

## Step 3: How sklearn actually decides which case applies

For one variable, you can literally check all three cases as above — solve
each, see which one is self-consistent, use that one.

For real datasets with many features, Lasso doesn't enumerate cases like
this by hand. sklearn uses **coordinate descent**: it updates one
coefficient at a time, holding the others fixed, using this exact same
soft-thresholding logic underneath. It's the same three-case check, just
applied automatically and repeatedly across every feature until convergence.
So the "which case is true" question isn't decided in advance — it's
discovered by solving each hypothesis and testing it, whether that's done
by hand (1 variable) or by the algorithm (many variables).

---

## Step 4: The factor of 2 (alpha vs alpha/2)

Differentiating the RSS term `Σ(Yi - mXi - b)²` genuinely produces a factor
of 2 (from the power rule) — that's not invented.

If the penalty is written as `alpha·|m|`, its derivative is `alpha·sign(m)`
— no 2 attached. So setting the full derivative to zero gives:

```
-2S + 2·D·m + alpha·sign(m) = 0
→ m = (S - alpha/2) / D     (for m>0 case)
```

Your teacher's version writes the penalty as `2·alpha·|m|` instead, so the
2's cancel cleanly:

```
-2S + 2·D·m + 2·alpha·sign(m) = 0
→ m = (S - alpha) / D
```

**This isn't bending math rules.** Alpha is just a free constant you choose
— nothing forces it to equal the "raw" penalty weight. Writing the penalty
as `2·alpha·|m|` is equivalent to defining `lambda = 2·alpha` and penalizing
by `lambda·|m|`. It's a relabeling of the same free parameter, done purely
so the constants cancel nicely in the final formula — not a mathematically
different model.

---

## Summary of the whole logic chain

1. `|m|` isn't differentiable at 0 → split into 3 sign-based cases.
2. Each case's formula is a **hypothesis**: solve it, then check its sign matches the assumption.
3. As alpha grows, the "valid" case's magnitude shrinks toward 0.
4. Once alpha ≥ |S|, both the m>0 and m<0 hypotheses become self-inconsistent → only m=0 remains valid.
5. This hard zero-point (not just shrinkage) is what makes Lasso produce **sparse** models, unlike Ridge.
6. The 2 in front of alpha is a convention (lambda = 2·alpha), not a broken derivation.
7. sklearn generalizes this exact logic via coordinate descent for multi-feature Lasso.
