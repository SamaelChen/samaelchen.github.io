---
title: 深度学习Numerical Computation
category: 深度学习
mathjax: true
date: 2017-01-25
---

深度学习Numerical Computation

<!-- more -->

1) Overflow and Underflow

The underflow occurs when number near zero rounded to zero. Lots functions behave differently when their argument is zero. For example, we want to avoid division by zero.

Overflow occurs when numbers is large, like $\infty$ or $- \infty$.

One example of a function stabilized against underflow and overflow is softmax:
$$
\text{softmax}(\boldsymbol{x})_i = \frac{\exp(x_i)}{\sum_{j=1}^n \exp(x_j)}
$$

Most packages have stabilized common numerically unstable expressions that arise in the context of deep learning.

2) Condition number

The condition number is an argument measures how much the output can change for a small change in the input arguments.

3) Gradient-Based Optimization

The function we want to minimize or maximize is the *objective function* or *criterion*. When we are minimizing it. We may call it the *cost function*, *loss function* or *error function*.

For example, $f(x - \varepsilon \text{sign}(f'(x)))$ is less than $f(x)$ for small enough $\epsilon$. We call this technique *gradient descent*. When $f'(x)=0$, the derivative provides no information about which direction to move.We define such points as *critical points* or *stationary points*. There are also some critical points neither *maxima* or *minima*. We call them *paddle points*.

We use partial derivative $\frac{\partial}{\partial x_i}f(\boldsymbol{x})$ to measure how $f$ changes as the variable $x_i$ increases at point $\boldsymbol{x}$. The gradient of $f$ is the vector containing all of the partial derivatives, denoted as $\bigtriangledown_{\boldsymbol{x}}f(\boldsymbol{x})$. In multiple dimensions, critical points are points where every element of the gradient is equal to zero.

The *directional derivative* in direction $\boldsymbol{u}$(a unit vector) is the slope of the function $f$ in direction $\boldsymbol{u}$. It's a projection of $\bigtriangledown_{\boldsymbol{x}}f(\boldsymbol{x})$ on $\boldsymbol{u}$. That means we can write it as
$$
\bigtriangledown_{\boldsymbol{u}}f(\boldsymbol{x}) = \lim \limits_{h \rightarrow 0} \frac{f(\boldsymbol{x}+h \boldsymbol{u})-f(\boldsymbol{x})}{h} = \boldsymbol{u}^{\top} \bigtriangledown_{\boldsymbol{x}} f(\boldsymbol{x})
$$

To minimize $f$, we would like to find the direction in which $f$ decrease the fastest. We do this like:
$$
\min \boldsymbol{u}^{\top} \bigtriangledown_{\boldsymbol{x}} f(\boldsymbol{x}) = \min \| \boldsymbol{u} \|_2 \|\bigtriangledown_{\boldsymbol{x}} f(\boldsymbol{x}) \|_2 \cos \theta
$$
where $\theta$ is the angle between $\boldsymbol{u}$ and the gradient. Substituting in $\| \boldsymbol{u} \|_2 = 1$, and we can simplifying it to $\min_{\boldsymbol{u}} \cos \theta$. We can move in the direction of the negative gradient to decrease $f$. We call this process as *method of steepest descent* or *gradient descent*:
$$
\boldsymbol{x}' = \boldsymbol{x} - \varepsilon \bigtriangledown_{\boldsymbol{x}} f(\boldsymbol{x})
$$
where $\varepsilon$ is the *learning rate*.

We can generalized gradient descent to discrete spaces. Ascending an objective function of discrete parameters is called *hill climbing*.

4) Beyond the Gradient

The matrix containing all partial derivatives is called *Jacobian matrix*. If we have a function $f: \mathbb{R}^m \rightarrow \mathbb{R}^n$, then the Jacobian matrix $J \in \mathbb{R}^{n \times m}$ of $f$ is defined such that $J_{i,j} = \frac{\partial}{\partial x_j} f(\boldsymbol{x})_i$.

The second derivative is $\frac{\partial^2}{\partial x_i \partial x_j} f$. In a single dimension, we can denote $\frac{d^2}{d x^2} f$ by $f''(x)$. The second derivative tells us how the first derivative will change as we vary the input. We can think of the second derivative as measuring *curvature*.

We collect all second derivatives into a matrix, and call it *Hessian matrix*.
$$
\boldsymbol{H}(f)(\boldsymbol{x})_{i,j} = \frac{\partial^2}{\partial x_i \partial x_j} f(\boldsymbol{x}).
$$

$H_{i,j} = H_{j,i}$ since $\frac{\partial^2}{\partial x_i \partial x_j} f(\boldsymbol{x}) = \frac{\partial^2}{\partial x_j \partial x_i} f(\boldsymbol{x})$ where the second derivatives are continuous. So the Hessian matrix is symmetric at such points.

The directional second derivative tells us how well we can expect a gradient descent step to perform. We can make a second-order Taylor series approximation to the function $f{\boldsymbol{x}}$ around the current point $\boldsymbol{x}^{(0)}$:
$$
f(\boldsymbol{x}) \approx f(\boldsymbol{x}^{(0)}) + (\boldsymbol{x} - \boldsymbol{x}^{(0)})^{\top} g + \frac{1}{2}(\boldsymbol{x} - \boldsymbol{x}^{(0)})^{\top} \boldsymbol{H} (\boldsymbol{x} - \boldsymbol{x}^{(0)}).
$$

If we use a learning rate of $\varepsilon$, then the new point $\boldsymbol{x}$ will be given by $\boldsymbol{x}^{(0)} - \varepsilon g$. Substituting this into our approximation:
$$
f(\boldsymbol{x}^{(0)} - \varepsilon g) \approx f(\boldsymbol{x}^{(0)}) - \varepsilon \boldsymbol{g}^{\top} \boldsymbol{g} + \frac{1}{2} \varepsilon^2 \boldsymbol{g}^{\top} \boldsymbol{H} \boldsymbol{g}.
$$

Let $\frac{\partial}{\partial \varepsilon} f(\boldsymbol{x}^{(0)} - \varepsilon \boldsymbol{g}) = 0$, we can get the best learning rate:
$$
\varepsilon = \frac{\boldsymbol{g}^{\top} \boldsymbol{g}}{\boldsymbol{g}^{\top} \boldsymbol{Hg}}
$$

In multiple dimensions, there is a different second derivative for each direction. That means in one direction, the derivative increases rapidly and in another direction, it may increase slowly. We have to choose a small step size to avoid overshooting the minimum and going uphill in directions with strong positive curvature. So the step size may be too small to make significant progress in other directions with less curvature.

This issue can be solved by using information from the Hessian matrix to guide the search. We called it *Newton's method*:
$$
f(\boldsymbol{x} + \triangle \boldsymbol{x}) \approx f(\boldsymbol{x}) + \triangle_{\boldsymbol{x}} \bigtriangledown f(\boldsymbol{x}) + \frac{1}{2} \triangle \boldsymbol{x}^{\top} \boldsymbol{H}(f)(\boldsymbol{x}) \triangle \boldsymbol{x} \\
\frac{\partial f(\boldsymbol{x} + \triangle \boldsymbol{x})}{\partial \triangle \boldsymbol{x}} = \bigtriangledown_{\boldsymbol{x}} f(\boldsymbol{x}) + \frac{1}{2} \boldsymbol{H}(f)(\boldsymbol{x}) \triangle \boldsymbol{x} + \frac{1}{2} \boldsymbol{H}^{\top}(f)(\boldsymbol{x}) \triangle \boldsymbol{x},
$$
where $\boldsymbol{H} = \boldsymbol{H}^{\top}$, we can get:
$$
\bigtriangledown f(\boldsymbol{x}) + \boldsymbol{H}(f)(\boldsymbol{x}) \triangle \boldsymbol{x} = 0 \\
\triangle \boldsymbol{x} = - \boldsymbol{H}^{-1}(f)(\boldsymbol{x}) \bigtriangledown_{\boldsymbol{x}} f(\boldsymbol{x}) \\
\boldsymbol{x}' = \boldsymbol{x} - \boldsymbol{H}^{-1}(f)(\boldsymbol{x}) \bigtriangledown_{\boldsymbol{x}} f(\boldsymbol{x}).
$$
