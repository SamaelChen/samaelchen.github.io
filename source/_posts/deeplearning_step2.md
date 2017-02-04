---
title: 深度学习概率论基础
categories: 深度学习
mathjax: true
date: 2017-01-17
---

深度学习概率论基础

<!-- more -->

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

8) Common Probability Distribution

8.1) Bernoulli Distribution

If $P(x=1)=p, P(x=0)=1-p, p(\text{x}=x)=p^x(1-p)^{1-x}$.

And we can get $\text{E}_{\text{x}}(x)=p, \text{Var}_{\text{x}}(x)=p(1-p)$.

It's a special *Binomial distribution*. We denote it as $X \sim B(n,p)$.

$f(x;n,p)=P(\text{x}=x)=C_n^x p^x(1-p)^{(n-x)}, \text{E}_{\text{x}}(x)=np, \text{Var}_{\text{x}}(x)=np(1-p)$.

8.2) Multinoulli Distribution
It's a special *Multinomial distribution*. The multinomial distribution is:
$$
\begin{align}
f(x_1, \dots, x_k; n ,p_1, \dots, p_k) &= P(\text{x}_1=x_1 \text{and } \dots \text{ and} \text{x}_n=x_n) \\
&=\begin{cases}
\frac{n!}{x_1! \dots x_k!} p_1^{x_1}\dots p_k^{x_k} &\text{when } \sum_{i=1}^k x_i=n \\
0 &\text{otherwise}
\end{cases}
\end{align}
$$

We can get this PMF from the following way:
$$
(p_1 + p_2 + \dots + p_k)=1 \\
\Downarrow \\
(p_1 + p_2 + \dots + p_k)^N = 1
$$

We choose $p_1$ $x_1$ times and $p_2$ $x_2$ times $\dots$ $p_k$ $x_k$ times. And $x_1+x_2+\dots+x_k=N$. It's like a problem about how to put $n$ balls into $k$ different boxes. Then we can get:
$$
C_n^{x_1} C_{n-x_1}^{x_2} \dots C_{n-x_1-\dots-x_{k-1}}^{x_k} = \frac{n!}{x_1!x_2! \dots x_k!}
$$

We can get the PMF. The expectation is $\text{E}_{\text{x}}(x_i)=np_i, \text{Var}_{\text{x}}(\text{X}_i) = np_i(1-p_i)$

And we can get
$$
\begin{align}
\text{Cov}(x_i, x_j) &= \frac{\text{Var}(x_i+x_j)-\text{Var}(x_i)-\text{Var}(x_j)}{2} \\
&= \frac{n(1-p_i-p_j)(p_i+p_j)-np_i(1-p_i)-np_j(1-p_j)}{2} \\
&= -n p_i p_j
\end{align}
$$

8.3) Gaussian Distribution

It's *normal distribution*:
$$
N(x;\mu,\sigma^2) = \sqrt{\frac{1}{2 \pi \sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)
$$
We can use precision $\beta = \sigma^{-2} \in (0, \infty)$ to replace $\sigma^2$.
$$
N(x;\mu,\beta^{-1}) = \sqrt{\frac{\beta}{2 \pi}} \exp(-\frac{1}{2} \beta (x - \mu)^2)
$$

The *multivariate normal distribution* is:
$$
N(\mathbf{x};\mathbf{\mu},\Sigma) = \sqrt{\frac{1}{(2 \pi)^2 \det(\Sigma)}} \exp(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^{\top}\Sigma^{-1}(\mathbf{x}-\mathbf{\mu}))
$$

Similarly, we can use *precision matrix* $\beta$:
$$
N(\mathbf{x};\mathbf{\mu}, \beta^{-1})=\sqrt{\frac{\det{\beta}}{(2 \pi)^n}} \exp(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^{\top} \beta (\mathbf{x}-\mathbf{\mu}))
$$

8.4) Exponential and Laplace Distributions

The exponential distribution has a sharp point at $x=0$.
$$
p(x;\lambda) = \lambda \mathbf{1}_{x \ge 0} \exp(-\lambda x)
$$

The indicator function $\mathbf{1}_{x \ge 0}$ assigns probability zero to all negative values of $x$.

The exponential distribution describes the time between events in a Poisson process(a process in which events occur continuously and independently at a constant average rate). And Poisson distribution is:
$$
p(N(t)=n) = \frac{(\lambda t)^n e^{-\lambda t}}{n!},
$$
where $N$ is a random variable, $t$ is the time, $n$ is the number of events. So when $n=0$:
$$
P(N(t)=0) = \frac{(\lambda t)^0 e^{-\lambda t}}{0!} = e^{-\lambda t}
$$
that means, when $k$-th event happened, in time $t$, the probability of $k+n$-th $(n=1, 2, 3, \dots)$ event is $1-e^{-\lambda t}$.

A related distribution is *Laplace distribution*
$$
Laplace(x; \mu, \gamma) = \frac{1}{2\gamma} \exp(-\frac{|x-\mu|}{\gamma})
$$

8.5) The Dirac Distribution and Empirical Distribution

The Dirac delta function can help us to specify that all the mass in a probability distribution clusters around a single point. The Dirac delta function is such a function that it's zero-valued everywhere except $0$, yet integrates to $1$. We define a PDF using it as:
$$
p(x) = \delta(x-\mu)
$$

A common use of the Dirac delta distribution is as a component of an *empirical distribution*,
$$
\hat{p}(\boldsymbol{x}) = \frac{1}{m} \sum_{i=1}^m \delta(\boldsymbol{x}-\boldsymbol{x}^{(i)}
$$
which puts probability mass $\frac{1}{m}$ on each of the $m$ points $\boldsymbol{x}^{(1)}, \dots \boldsymbol{x}^{(m)}$ forming a given data set or collection of samples.

The Dirac delta distribution is define the empirical distribution over continuous variables. For discrete variables, we can conceptualize an empirical distribution as a multinoulli distribution.

8.6) Mixtures of Distribution

One common way of combining distributions is to construct a *mixture distribution*.

A mixture distribution contains lots component distributions. On each trial, sampling a component identity from a multinoulli distribution determine the choice of which component distribution generates the sample:
$$
P(x)=\sum_i p(c=i)p(x|c=i)
$$

The mixture model allow us to briefly glimpse a concept of the *latent variable*. A latent variable is a random variable that we cannot observe directly.

A powerful and common mixture model is *Gaussian mixture* model, in which the components $p(\boldsymbol{x}|c=i)$ are Gaussian.

9) Useful Properties of Common Functions

+ logistic sigmoid:

$$
\sigma(x) = \frac{1}{1+\exp(-x)}
$$
This function becomes flat and insensitive to small changes in its input when its argument is positive infinity or negative infinity.

+ softplus function:

$$
\zeta(x) = \log(1+\exp(x))
$$

The following properties are all useful:
$$
\begin{align}
\sigma(x) &= \frac{\exp(x)}{1+\exp(x)} \\
\frac{d}{dx} \sigma(x) &= \sigma(x)(1-\sigma(x)) \\
1-\sigma(x) &= \sigma(-x) \\
\log(\sigma(x)) &= -\zeta(-x) \\
\frac{d}{dx} \zeta(x) &= \sigma(x) \\
\forall x \in (1,0), &\sigma^{-1}(x) = \log(\frac{x}{1-x}) \\
\forall x > 0, &\zeta^{-1}(x) = \log(\exp(x)-1) \\
\zeta(x) &= \int_{-\infty}^x \sigma(y)dy \\
\zeta(x) - \zeta(-x) &= x
\end{align}
$$

10) Bayes' Rule

Bayes' rule:
$$
P(\text{x}|y) = \frac{P(\text{x})P(y|\text{x})}{P(y)}
$$
And $P(y) = \sum \limits_{\text{x}} P(y|x)P(x)$.

11) Information Theory

We wanna quantify information like below:
+ Similar events should have low information content, and in the extreme case, events that are guaranteed to happen should have no information content whatsoever.
+ Less likely events should have higher information content.
+ Independent events should have additive information.

We define the *self-information* of an event $\text{x} = x$ to be:
$$
\text{I}(x) = -\log P(x)
$$
The units of I(x) is *nats*. One nat is the amount of information gained by observing an event of probability $\frac{1}{e}$. If we use base-$2$ logarithms and units called *bits* or *shannons*

We can quantify the amount of uncertainty in an entire probability distribution using the *Shannon Entropy*:
$$
H(\text{x}) = \sum_{i=1}^n P(x_i)I(x_i) = - \sum_{i=1}^n P(x_i)\log_b P(x_i)
$$
where $b = 2 \text{ or } e \text{ or } 10$.

If we have two separate probability distributions $P(\text{x})$ and $Q(\text{x})$ over the same random variable $\text{x}$, we can measure how different these two distributions are using
the *Kullback-Leibler (KL) divergence*:
$$
D_{KL}(P \| Q) = E_{x \sim P} \Big[ log \frac{P(x)}{Q(x)} \Big] = E_{x \sim P}[\log P(x) - \log Q(x)]
$$

The KL divergence is 0 if and only if P and Q are the same distribution in the case of discrete variables, or equal "almost everywhere" in the case of continuous variables. However, it is not a true distance measure because it is not symmetric: $D_{KL}(P \| Q) \ne D_{KL}(Q \| P)$ for some $P$ and $Q$.

A quantity that is closely related to the KL divergence is the *cross-entropy* $H (P, Q) = H(P) + D_{KL}(P \| Q)$, which is similar to the KL divergence but lacking
the term on the left:
$$
H(P, Q) = -E_{x \sim P} \log Q(x).
$$

12) Structured Probabilistic Models

Probabilistic Graphical Models?! It's too hard for me right now. I'll finish it in the future.
