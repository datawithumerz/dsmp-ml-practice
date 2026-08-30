# Lasso vs Ridge — Why Lasso Creates Sparsity (Full Notes)

This covers three angles on the same result: the **algebra** (why the formula
zeroes out), the **geometry** (why the diamond shape causes it), and the
**shrinkage speed** (why Ridge and Lasso shrink differently as alpha grows).

---

## Part 1: The algebraic derivation

### Setup

For simple linear regression with Lasso penalty:

```
Loss = Σ(Yi - (mXi + b))² + alpha·|m|
```

Let `S = Σ(Yi - Y_mean)(Xi - X_mean)` and `D = Σ(Xi - X_mean)²`.
`S/D` is just the plain OLS slope (no regularization).

`|m|` isn't differentiable at m=0, so we can't take one derivative for all
m — we split into cases.

### The three-case check

Each formula below is a **hypothesis**: solve it assuming a sign, then check
if the result actually has that sign. Only a self-consistent case is used.

- **m > 0 hypothesis:** `m = (S - alpha) / D` — valid only if this is > 0
- **m < 0 hypothesis:** `m = (S + alpha) / D` — valid only if this is < 0
- **m = 0:** used when neither of the above is self-consistent

### Why increasing alpha kills m

Assume S > 0. As alpha grows, `(S - alpha)/D` shrinks toward 0. Once
`alpha = S`, m becomes 0. Push alpha further and this formula would go
negative — but that violates its own "m>0" assumption, so it's rejected.
The m<0 formula also fails its own assumption (it comes out positive, not
negative). With **both hypotheses invalid**, the only value left is m=0.

**General threshold** (covers both signs of S, since alpha ≥ 0 always):

```
m = 0   whenever   alpha ≥ |S|
```

This "snap to exactly zero past a threshold" is called **soft-thresholding**
— it's the core mechanism of Lasso sparsity. Ridge has no such threshold
because its penalty (m²) is smooth everywhere, with no non-differentiable
point at 0.

### The factor of 2 (alpha vs alpha/2)

Differentiating the RSS term genuinely produces a factor of 2. If the
penalty is written as `alpha·|m|` (no built-in 2), the true derivative gives
`m = (S - alpha/2)/D`. Writing the penalty as `2·alpha·|m|` instead makes
the 2's cancel, giving the cleaner `m = (S - alpha)/D`.

This isn't "breaking math" — alpha is just a constant you choose. Defining
`lambda = 2·alpha` and penalizing by `lambda·|m|` is the exact same model,
just with the constant relabeled so the formula looks cleaner.

### How sklearn generalizes this

For one variable, you can literally check the three cases by hand. For many
features, sklearn's Lasso uses **coordinate descent**: it updates one
coefficient at a time using this same soft-thresholding logic, repeated
across all features until convergence. Same idea, automated.

---

## Part 2: The geometric intuition

### The picture (Figure 6.7, ISLR)

- Red contours = level curves of RSS (all points on one ring have equal
  error; error grows on the outer rings). It's the top-down view of the
  error "bowl," centered at the OLS solution β̂.
- Blue shape = the constraint region: a **diamond** for Lasso
  (`|β1|+|β2| ≤ s`), a **circle** for Ridge (`β1²+β2² ≤ s`).
- The solution is wherever the smallest RSS contour first touches the
  boundary of the blue shape.

### The goat-and-rope analogy

Imagine a goat tied to a peg at the origin with a rope of fixed length
(`s`, the budget). The goat wants to reach the best grass (β̂), but the rope
limits how far it can go.

**Ridge = an open, round field.** The rope traces a perfect circle. Every
point on the boundary looks the same — smooth, no special spot. Wherever
the rope runs out, it's just... wherever. Nothing pulls it toward a
particular axis.

**Lasso = a diamond-shaped courtyard with sharp corners.** The corners sit
*exactly on the axes* (e.g. the corner at (s, 0) has the second coordinate
equal to exactly 0). Corners stick out further than the flat walls next to
them — they're the closest reachable points in many directions. So when the
goat pulls in some tilted direction (toward β̂), the rope very often runs
out exactly at a corner rather than along a flat wall.

**Landing on a corner means:** one direction is exactly 0 — the goat isn't
moving in that direction at all.

### The takeaway

A round boundary (Ridge) has no special corner to land on, so the solution
ends up close to zero in each direction but not exactly zero. A diamond
boundary (Lasso) has corners sitting right on the zero-lines, and solutions
very often land exactly there — forcing a coefficient to be exactly zero.

Same conclusion as the algebra above (soft-thresholding), just seen through
shapes touching instead of equations being solved.

---

## Part 3: Why Ridge shrinks faster at first, but Lasso "wins" eventually

Using the coordinate-wise closed forms (axis-aligned case, b = OLS
coefficient, a = curvature of RSS in that direction):

```
Lasso:  β = max(b - alpha/(2a), 0)      → straight line, constant slope
Ridge:  β = a·b / (a + alpha)           → curve, steep at first, flattens
```

**Lasso's shrinkage is a constant tug** — it subtracts the *same fixed
amount* per unit of alpha, no matter how big alpha already is. Like paying
a flat toll every step: steady, linear, and once the balance hits zero, it
just stops there.

**Ridge's shrinkage is a multiplying squeeze** — dividing by a growing
denominator. Near alpha=0, adding even a little to the denominator changes
the ratio a lot in percentage terms, so Ridge coefficients drop *fast*
initially. But as alpha keeps growing, each additional unit matters less
and less (diminishing returns) — it keeps approaching zero but never
actually touches it.

**Everyday analogy:** Ridge is like your income being divided by a growing
tax rate — the sting is big early on, but it never actually reduces you to
zero. Lasso is like paying a flat monthly fee — steady drain, and once your
balance hits zero, it's just zero, done.

So: Ridge can *look* faster early on, but it never reaches exact zero.
Lasso's steady, constant subtraction is what eventually hits the wall and
clips a coefficient to exactly 0 — which is the sparsity Ridge can never
produce.

---

## One-line interview answer

"Lasso's penalty isn't differentiable at zero, so the optimum has to be
checked as separate sign-based cases; past a threshold on alpha, both the
positive and negative cases become inconsistent, leaving only m=0 —
geometrically, this shows up because the L1 constraint region has corners
sitting exactly on the axes, and the expanding RSS contours tend to touch
those corners first, unlike Ridge's smooth circular region which has no
such special point."
