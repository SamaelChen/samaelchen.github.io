---
title: 台大李宏毅机器学习 01
category: 统计学习
mathjax: true
date: 2017-08-26
---

Machine learning is finding a function to solve problems. Regression is such a method to predict a scalar.
<!-- more -->

Here we wanna predict the CP value of a Pokemon, we can use $x$ to represent it. We use $x_{cp}$ to represent the CP value of it, and $x^i$ as the $i$th Pokemon.

If we wanna find a function to fit this data, we can follow these steps:

Step1. Assume the real function is $y = b + w \cdot x_{cp}$, and the set $\{f_1, f_2, \dots \}$ is our models. Theoretically, there exists infinite model. This function assume the CP value is effected by and only by CP value. In general, we assume $y = b + \sum w_i \cdot x_i$, $x_i$ as all possible features.

Step2. We should find a method to measure how good the model we assumed is. In this case, we can use MSE(Mean Square error). It is defined as $MSE = \sum (y_i - \hat{y}_i)^2$. I used to represent the real value as $y$, and the predict value as $\hat{y}$, this is a little different with the course. There are lot of functions like MSE, we can call these functions as "Loss function". And the MSE of step1 can be wrote as
$$L(f) = \sum (y^n - f(x^n))^2 = \sum (y^n - (w \cdot x^n + b))^2.$$

Step3. Now we have a set of hypothesis and a loss function to measure them, we can find out the best function $f$. In this case, we wanna minimize the loss function, and we can represent it as
$$\arg \min L(w, b) = \sum (y^n - (w \cdot x^n + b))^2.$$
We use a general method, Gradient Descent, to solve it. Consider loss function $L(w, b)$ with one parameter $w$.

+ First, we randomly pick an initial value $w^0$
+ Second, we compute $\frac{\partial{L}}{\partial{w}}|_{w=w^0}$.
+ Third, update $w$. $w^1 = w^0 - \eta \frac{\partial{L}}{\partial{w}}|_{w=w^0}$, $\eta$ as learning rate.
+ Repeat these steps until $\frac{\partial{L}}{\partial{w}}|_{w=w^n} = 0$.

But this method is greedy, it may stuck at saddle point or local minimum not global minimum. In general, this situation only happen when loss function is complex.

How about multiple parameters? We can write it as
$$
\bigtriangledown L = \begin{bmatrix}
\frac{\partial{L}}{\partial{w}} \\
\frac{\partial{L}}{\partial{b}} \\
\end{bmatrix}_{\text{gradient}}.
$$

In this case, professor Li got $\text{error}_{\text{training}} = 31.9$ and $\text{error}_{\text{test}} = 35$.

We've discussed the simplest model, how about increase the complexity of our model? If we assume $y = b + w_1 \cdot x + w_2 \cdot x^2$, we will get $\text{error}_{\text{training}} = 15.4, \ \text{error}_{\text{test}} = 18.4$, it looks better. If we assume $y = b + w_1 \cdot x + w_2 \cdot x^2 + w_3 \cdot x^3$, and $\text{error}_{\text{training}} = 15.3,\  \text{error}_{\text{test}} = 18.1$.
And degree of $x$ is 4, $\text{error}_{\text{training}} = 14.9, \ \text{error}_{\text{test}} = 28.8$. If $y = b + w_1 \cdot x + w_2 + \cdot x^2 + w_3 \cdot x^3 + w_4 \cdot x^4 + w_5 \cdot x^5$, $\text{error}_{\text{training}} = 12.8, \ \text{error}_{\text{test}} = 232.1$.

When the complexity increase, the training error decrease and the testing error decrease then increase. If the difference between training error and testing error is very big, we call it **overfitting**.

How to deal with overfitting? We can use domain knowledge to design our model and don't design it too complex. The other practical way is collecting more data, and add more features.

If we collect four different Pokemons, we can design a model:
$$
\begin{align}
y &= b_1 \cdot \delta(x_s=\text{Pidgey}) \\
&+ w_1 \cdot \delta(x_s = \text{Pidgey})x_{cp} \\
&+ b_2 \cdot \delta(x_s = \text{Weedle}) \\
&+ w_2 \cdot \delta(x_s = \text{Weedle})x_{cp} \\
&+ b_3 \cdot \delta(x_s = \text{Caterpie}) \\
&+ w_3 \cdot \delta(x_s = \text{Caterpie})x_{cp} \\
&+ b_4 \cdot \delta(x_s = \text{Eevee}) \\
&+ w_4 \cdot \delta(x_s = \text{Eevee})x_{cp}
\end{align}
$$
where
$$
\delta(x_s = Pidgey) = \begin{cases}
=1 \quad \text{if } x_s = \text{Pidgey} \\
=0 \quad \text{otherwise}
\end{cases}.
$$

Now we can get $\text{error}_{\text{training}} = 3.8, \; \text{error}_{\text{test}} = 14.3$.

If we add more factors and increase the complexity of our model, we may cause overfitting again. And now we can redesign our loss function as
$$
L = \sum_n \Big(y^n - \big( b + \sum w_i x_i \big) \Big)^2 + \lambda \sum (w_i)^2.
$$
The functions with smaller $w_i$ are better, and smaller $w_i$ means we will get a smoother function. We believe smoother function is more likely to be correct. We call this method regularization. Note that, we don't need to apply regularization on $b$, since it just make function move up and down.
