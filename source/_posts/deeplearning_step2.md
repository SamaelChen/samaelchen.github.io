---
title: 深度学习概率论基础
categories: 深度学习
mathjax: true
---
1) We use probability to represent a *degree of belief*. The probability relates to the rates at which events occur is *frequentist probability*, while relates to qualitative levels of certainty is *Bayesian probability*.

2) *Random variable* is a variable that can take on different values randomly. Random variable may be discrete or continuous. We use $\textrm{x}$ to denote it, and $x$ as one value.

3) Probability Distributions

Probability distributions is a description of how possibly a random variable or set of random variables is to take on each of its possible states.

3.1) Discrete Variables and Probability Mass Functions

We denote probability mass functions with a capital $P$. It maps from a state of a random variable to the probability of that random variable taking on that state. We write it as $P(\textrm{x}=x)$, and we can also write it as $P(x)$. Similarly, the *joint probability distribution* is $P(x, y)$.

The function $P$ has the following properties:
+ The domain of $P$ must be the set of all possible states of $\text{x}$.
+ $\forall x \in \text{x}, 0 \le P \le 1$
+ $\sum_{x \in \text{x}} P(x) = 1$. We refer to this property as being *normalized*.

3.2) Continuous Variables and Probability Density Functions

We use $p$ to represent probability density functions. It has the following properties:
+ The domain of $p$ must be the set of all possible states of $\text{x}$.
+ $\forall x \in \text{x}, p(x) \ge 0$. We do not require $p(x) \le 1$
+ $\int p(x) dx = 1$

A probability function does not give the probability of a specific state directly, it's $p(x) \delta x$ with volume $\delta x$ landing inside an infinitesimal region.

We can integrate the density function to find the actual probability mass of a set of points.

4) Marginal Probability

The probability over the subset is *marginal probability* distribution. For example:
$$
\forall x \in \text{x}, P(\text{x}=x) = \sum \limits_{y}P(\text{x}=x, \text{y}=y)
$$

For continuous variables, we need to use integration instead of summation:
$$
p(x) = \int p(x,y)dy
$$

5) The Conditional Probability and Chain Rule

$$
P(\text{y}=y|\text{x}=x) = \frac{P(\text{y}=y, \text{x}=x)}{P(\text{x}=x)}
$$

$$
P(\text{x}^{(1)}, \dots, \text{x}^{(n)}) = P(\text{x}^{(1)}) \prod_{i=1}^n P(\text{x}^{(i)}|\text{x}^{(1)}, \dots, \text{x}^{(n)})
$$
For example:
$$
\begin{align}
P(a, b, c) &= P(a|b,c)P(b,c)\\
P(b,c) &= P(b|c) P(c) \\
P(a, b, c) &= P(a|b,c)P(b|c)P(c)
\end{align}
$$

6) Independence and Conditional Independence

Two random variables $\text{x}$ and $\text{y}$, if their probability distribution is a product of two factors:
$$
\forall x \in \text{x}, y \in \text{y}, P(\text{x}=x, \text{y}=y) = P(x)P(y),
$$
they are independent.

The conditional independent is:
$$
\forall x \in \text{x}, y \in \text{y}, z \in \text{x}, P(\text{x}=x, \text{y}=y|\text{z}=z) = P(x|\text{z}=z)P(y|\text{z}=z).
$$
We can denote it as $\text{x} \bot \text{y}$ or $\text{x} \bot \text{y}|\text{z}$.

7) Expectation, Variance and Covariance

Expectation:
$$
\begin{cases}
E_{x \sim p[f(x)]} &= &\sum \limits_x P(x)f(x), \text{for discrete variables} \\
E_{x \sim p[f(x)]} &= &\int p(x)f(x)dx, \text{for continuous variables}
\end{cases}
$$
Expectations are linear:
$$
E_{\text{x}}[\alpha f(x) + \beta g(x)] = \alpha E_{\text{x}}(f(x)) + \beta E_{\text{x}}(g(x))
$$

The variance is:
$$
Var(f(x)) = E[(f(x)-E[f(x)])^2] = E(x^2) - E(x)^2 \\
Sd[f(x)] = \sqrt{Var(f(x))}
$$

The covariance is:
$$
Cov(f(x), g(y)) = E[(f(x)-E(f(x)))(g(y)-E(g(y)))] = E(f(x)g(y))-E(f(x))E(g(y))
$$
And correlation is:
$$
Cor(f(x), g(y)) = \frac{Cov(f(x), g(x))}{Sd(f(x))Sd(g(y))}
$$
If $x$ and $y$ are independent, their $Cov$ must be zero. But if their $Cov = 0$, they may not independent.

For instant, $x \sim U(-1,1)$ and a random variable $s$. With probability $\frac{1}{2}$, we choose $s=1$, otherwise $s=-1$. We generate a random variable $y=sx$.
$$
\begin{align}
Cov(x,y) &= E(xy)-E(x)E(y) \\
&= E(sx^2)-0 \\
&= E(s)E(x^2) \\
&= 0
\end{align},
$$
but $x$ and $y$ are not independent.
