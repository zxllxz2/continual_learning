---
layout: post
title: "EWC"
description: Introduction to EWC
---



Motivation for Elastic Weight Consolidation (EWC)
============
Although L-2 norm regularization moderates catastrophic forgetting in some sense, it has one serious problem: no distinction in feature importance
of previous tasks. As a result, L2-norm regularization may pose great restrictions for all features, and, overall, the restriction can be so severe that the
neural network can only remember previous tasks at the expense of not learning the new task. In light of this situation,
elastic weight consolidation (EWC) comes to the rescue: EWC is able to distinguish between important and unimportant features, and will
penalize features that are critical to previous tasks severely while penalizing marginal features slightly. This allows simultaneous remembering and learning.



Idea behind EWC
============

EWC tackles the problem from a probabilistic perspective. Assume that we are trying to continually learn from a collection of datasets, D. The
conditional probability that we are trying to optimize would be *<span>log p(&theta; | D)</span>*. Let's first consider the two-task case.

Suppose *<span>D</span>* is comprised of independent and disjoint datasets *<span>D<sub>A</sub></span>* and
*<span>D<sub>B</sub></span>*, and it follows that *<span>D = D<sub>A</sub> ∪ D<sub>B</sub></span>*. For the 
two-task case, the conditional probability *<span>log p(&theta; | D)</span>* is equivalent to *<span>log p(&theta; | D<sub>A</sub> + D<sub>B</sub>)</span>*.
Using Beyes' rule, we can compute *<span>log p(&theta; | D)</span>* using the following expression:

<p align="center">
  <img width="320" height="38" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/Bayes_rule_eq2.jpg?raw=true">
</p>

*<span>log p(&theta; | D)</span>* is the posterior of continually learning two tasks, and terms in the above expression
corresponds to the negative loss of the second task, prior of the second task (also posterior of the first task),
and the normalization respectively. It can be easily inferred that all information about previous task should be contained
in the term *<span>log p(&theta; | D<sub>A</sub>)</span>*. In order to perform maximum a posterior (MAP) method, we need to find 
a way to represent the posterior of the previous task, *<span>log p(&theta; | D<sub>A</sub>)</span>*. Nevertheless, the exact posterior is intractable and 
we do not have access to data of previous tasks, so it must be approximated cleverly. One way to achieve this is through Laplace 
Approximation, which will be discussed briefly here.

The crux of Laplace approximation is the second-degree Taylor expansion. Denote *<span> h(&theta;) = log p(&theta; | D<sub>A</sub>)</span>*, and let *<span>&theta;*</span>* be the point where *<span>h(&theta;)</span>*
is optimum. Second degree Taylor expansion would give us an approximation of *<span>h(&theta;)</span>*:

<p align="center">
  <img width="350" height="50" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/Taylor_expansion_eq3.jpg?raw=true">
</p>


The first term is a constant and the second term zero. Hence, the approximation can be simplified
to the following using Hessian matrix:

<p align="center">
  <img width="350" height="50" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/Hessian_approximation_eq4.jpg?raw=true">
</p>

Laplace approximation can be a solid approximation for the feature importance. However, the involvement
of the second derivative makes it hard to implement in practice. A further approximation for the Hessian
matrix is then needed for the posterior. The approximation we'll choose is the Fisher Information Matrix (FIM).
FIM is defined to be the matrix multiplication of the first derivative, and, in our context, the FIM can be computed
as

<p align="center">
  <img width="230" height="47" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/FIM_eq5.jpg?raw=true">
</p>

FIM has three properties: *<span>(i)</span>* It is equivalent to the second derivative of teh loss near the 
minimum, *<span>(ii)</span>* it can be computed from first-order derivative alone, and *<span>(iii)</span>* it 
is guaranteed to be positive semi-definite. Based on these, the Hessian matrix can then be approximated by *<span>-F</span>*.
This provides a further approximation for the posterior:

<p align="center">
  <img width="300" height="47" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/FIM_approximation_eq6.jpg?raw=true">
</p>

If we define a hyper-parameter *<span>&lambda;</span>* that determines the importance of the old task compared with the new one, 
MAP then gives the loss function *<span>L</span>* that we should minimize in EWC for two-task case:

<p align="center">
  <img width="230" height="50" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/EWC_loss_eq7.jpg?raw=true">
</p>


Offline EWC
============

Obviously, it is uncommon for real world to contain only two tasks. So, now, we will step into multi-task continual learning.
Offline EWC is the first multi-task technique we'll explore.

Offline EWC is a natural extension of the two-task EWC. It strictly follows the idea of EWC by storing all fisher information matrices from previous tasks,
and adding them one by one as the regularization term when learning a new task. Suppose we are trying to learn the *<span>K<sup>th</sup></span>* task, the 
loss function *<span>L</span>* using offline EWC would be

<p align="center">
  <img width="300" height="65" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss_offline_EWC_eq8.jpg?raw=true">
</p>

Typically, the *<span>&lambda;</span>* value used for each task will be set to be the same value. However, it is of no cost to set *<span>&lambda;</span>*
individually for particular uses.


Implementation of offline EWC
============

The offline EWC is implemented below using pytorch

~~~python
class OfflineEWC:
    def __init__(self, model: nn.Module, loss=nn.MSELoss()):
        self._model = model

        self._params = []
        self._fims = []
        self._loss = loss
        self._optim = None
        # self._lambda = []

    def train(self, inputs, labels, lam, lr=8e8, epochs=500):

        self._optim = torch.optim.Adam(self._model.parameters(), lr=lr)

        loss_values_x1 = []

        # First training period
        for _ in range(epochs):

            f = self._model(inputs.float())

            regularizer = 0

            for n, p in self._model.named_parameters():
                for i in range(len(self._fims)):
                    regularizer += torch.dot(self._fims[i][n].reshape(-1), ((p - self._params[i][n]) ** 2).reshape(-1))

            loss = self._loss(f, labels.unsqueeze(1).float()) + lam * regularizer
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            # calculate and store the loss per epoch for both datasets
            loss_values_x1.append(loss.item())

        self._params.append({})
        temp_param = {n: p for n, p in self._model.named_parameters() if p.requires_grad}
        for n, p in deepcopy(temp_param).items():
            self._params[-1][n] = p

        f = self._model(inputs.float())
        loss = self._loss(f, labels.unsqueeze(1).float())
        self._optim.zero_grad()
        loss.backward()

        temp_fisher = {}
        for n, p in self._model.named_parameters():
            temp_fisher[n] = p.grad.data

        self._fims.append({})
        for n, p in temp_fisher.items():
            self._fims[-1][n] = p ** 2

        return loss_values_x1
~~~

Demo of offline EWC
============

Next, we will try to convince you that offline EWC works through an example of four individual tasks. The data on which we're trying to train
continually is the following, and we will be using a 4-hidden-layer MLP with perceptron number of 1, 100, 100, 100, 100, and 1.

![offline4_data](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/data_online4.png?raw=true)

Below is the trace of the experiments after each individual task being trained

Task 1:

![loss1_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss1_offline4.png?raw=true)
![task1_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task1_offline4.png?raw=true)

Task 2:

![loss2_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss2_offline4.png?raw=true)
![task2_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task2_offline4.png?raw=true)

Task 3:

![loss3_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss3_offline4.png?raw=true)
![task3_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task3_offline4.png?raw=true)

Task 4:

![loss4_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss4_offline4.png?raw=true)
![task4_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task4_offline4.png?raw=true)


What can be improved?
============

The advantage of using offline EWC is obvious: it alleviates the problem of catastrophic forgetting and mimic the effect
of Hessian matrix to the greatest degree. However, its downside can also be annoying. Imagine a situation such that there are hundreds of thousands
tasks waiting to be learnt. Offline EWC will perform poorly since it tries to store fisher information matrix for each task being
learnt, and there will be hundreds of thousands of them. So, in this case, not only the space consumption will be large, but also the
computation cost wil be huge.

What can we do then? Here is a hint: can we reduce the number of information needed? The next section would give you the answer.
