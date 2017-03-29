---
title: 遗传算法
category: 统计学习
mathjax: true
date: 2017-03-29
---

遗传算法的简要实现

<!-- more -->

遗传算法的理论基础其实很简单，将每一个解当作种群中的一个个体，足够多的解构成一个种群。然后各种迭代，迭代过程中随机替换两个解中的元素，模拟基因重组；或者按照一定概率变异，改变一个解中的某个元素，模拟基因突变。然后适者生存就找到最优解。

整个算法的理解上没啥数学的东西，不懂为什么能把这玩意儿说明白的就能被当作大神。另外，如果不是正经学生物的，别太较真，反正我是不能理解单倍体生物在有丝分裂的时候怎么实现交叉的。另外需要注意的是，为了提高寻优的效率，突变概率，交叉概率都比自然界中真实的概率高多了。

所以实现上基本上可以拆成以下几部分的功能：

+ 生成种群

+ 基因重组

+ 突变

+ 适者生存

这里需要注意，适者生存有多种实现方法，最简单的例如每次取前10%。稍微符合自然规律的可以使用一些轮盘法啥的，简单理解，其实同性恋在野生环境下是不具有任何生存优势的，因为同性无法产生后代。但是经过几万年的演化，仍然有同性恋存在，只是比例比较少而已。

> 生成种群
```python
def generatepop(lowerbound, upperbound, popsize=100):
    # 生成种群
    # lowerbound, upperbound很好理解，设定值域
    # popsize设定种群中个体数量
    pop = []
    for i in range(popsize):
        vec = [random.randint(lowerbound[j], upperbound[j])
               for j in range(len(lowerbound))]
        pop.append(vec)
    return pop
```

> 基因重组
```python
def crossover(gene1, gene2):
    # 基因重组
    i = random.randint(1, len(gene1) - 2)
    return gene1[0:i] + gene2[i:]
```

> 基因突变
```python
def mutate(gene, lowerbound, upperbound, mutation_prob):
    # 基因突变
    for i in range(len(gene)):
        tmp = random.randint(lowerbound[i], upperbound[i])
        if random.random() > mutation_prob[i] and gene[i] != tmp:
            gene[i] = tmp
    return gene
```

> 主函数（这里适者生存使用了最简单的排序取最优）
```python
def geneticoptimize(lowerbound, upperbound, cost,
                    crossover_prob=0.3, elite=0.1, maxiter=100,
                    popsize=100, argmin=True):
    populations = generatepop(lowerbound, upperbound, popsize)
    # mutation_prob是一个list，表示每个DNA突变概率不同
    mutation_prob = [random.random() * 0.8 for i in range(len(lowerbound))]
    for i in range(maxiter):
        scores = [(cost(v), v) for v in populations]
        if argmin:
            scores.sort()
        else:
            scores.sort(reverse=True)
        ranked = [v for (s, v) in scores]
        # 适者生存
        elites_size = int(elite * popsize)
        populations = ranked[0:elites_size]
        # 开始产生后代
        while len(populations) < popsize:
            if random.random() > crossover_prob:
                c1 = random.randint(0, elites_size)
                c2 = random.randint(0, elites_size)
                populations.append(crossover(ranked[c1], ranked[c2]))
            else:
                c = random.randint(0, elites_size)
                populations.append(mutate(ranked[c], lowerbound,
                                          upperbound, mutation_prob))
        print(scores[0][0])
    return scores[0][1]
```

详细可以看我写的渣[code](https://github.com/SamaelChen/hexo-practice-code/blob/master/sp/optimization/genetic_algorithm.py)。
