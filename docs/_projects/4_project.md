---
layout: post
title: "Online EWC"
description: Introduction to Online EWC
---
<!-- Example modified from [here](http://www.unexpected-vortices.com/sw/rippledoc/quick-markdown-example.html){:target="_blank"}. -->

Motivation for Online EWC
============

Last section mentions that space and time complexity of Offline EWC can become unacceptable as task number grows.
In light of this, Online EWC is introduced as a variant of the EWC technique. Online EWC compromises the 
performance for a better complexity than the Offline version. So, it makes sense considering Online EWC as 
a product of the trade-off between performance and complexity.



How Online EWC works
--------------


Online EWC realizes multi-task continual learning by maintaining a single FIM (call it online
FIM for differentiating purpose). This online FIM gets updated each time a new task is trained. Denote the online
FIM before the update as *<span>F<sub>old</sub></span>* and the online FIM after the
update as *<span>F<sub>new</sub></span>*. Let *<span>F<sub>c</sub></span>* be the FIM corresponding to the
current task, and &alpha; be the importance coefficient controlling the weight of previous tasks. The update process of the
online FIM can then be formulated as follows:

<p align="center">
  <img width="250" height="38" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/Online_FIM_eq9.jpg?raw=true">
</p>

Given the maintenance of a single FIM, suppose we are trying to learn the *<span>K<sup>th</sup></span>* task, the loss function *<span>L</span>* using Online
EWC would then be

<p align="center">
  <img width="250" height="47" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/Online_Update_eq10.jpg?raw=true">
</p>

Implementation of Online EWC
--------------

Below we show our implementation of Online EWC using pytorch

~~~python
class OnlineEWC:
    def __init__(self, model: nn.Module, loss=nn.MSELoss()):
        self._model = model
        self._params = {}
        self._fim = {}
        self._loss = loss
        self._loss_lst = {}
        self._optim = None
        self._lambda = 0
        self._time = 0

    def train(self, inputs, labels, index, lr, alpha = 0.5, lam=0, epochs=500):
        self._loss_lst = {}
        self._optim = torch.optim.Adam(self._model.parameters(), lr=lr)

        loss_values_x1 = []
        self._lambda = lam
        self._time = 0

        # training
        for _ in range(epochs):
            start_time = time.time()
            f = self._model(inputs[index].float())
            regularizer = 0
            if len(self._params) != 0:
                loss_ewc = 0
                for n, p in self._model.named_parameters():
                    loss_ewc += torch.matmul(self._fim[n].T, (torch.reshape(p, (-1,1)) - torch.reshape(self._params[n], (-1,1))) ** 2)
                regularizer += self._lambda * loss_ewc

            loss = self._loss(f, labels[index].unsqueeze(1).float()) + regularizer
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
            self._time += time.time() - start_time

            # store loss
            loss_values_x1.append(loss.item())

            if index in self._loss_lst:
                self._loss_lst[index].append(loss_values_x1[-1])
            else:
                self._loss_lst[index] = [loss_values_x1[-1]]

            for i in range(len(inputs)):
                if i != index:
                    tmp_f = self._model(inputs[i].float())
                    tmp_loss = self._loss(tmp_f, labels[i].unsqueeze(1).float())
                    if i in self._loss_lst:
                        self._loss_lst[i].append(tmp_loss)
                    else:
                        self._loss_lst[i] = [tmp_loss]

        start_time = time.time()
        for n, p in deepcopy(self._model).named_parameters():
            if p.requires_grad:
                self._params[n] = p

        # update fisher information matrix
        f = self._model(inputs[index].float())
        loss = self._loss(f, labels[index].unsqueeze(1).float())
        self._optim.zero_grad()
        loss.backward()

        temp_fisher = {}
        for n, p in self._model.named_parameters():
            temp_fisher[n] = torch.reshape(p.grad.data, (-1,1))

        for n, p in temp_fisher.items():
            if n in self._fim:
                self._fim[n] = self._fim[n]*alpha + p**2 * (1-alpha)
            else:
                self._fim[n] = p**2
        self._time += time.time() - start_time
~~~

To compare online EWC with offline EWC, it's a good idea to conduct experiments on online EWC with the same sample data 
as that of offline EWC. The sample data we use is as follows

![online4_data](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/data_online4.png?raw=true)

Just like what we did for the Offline EWC, we use a 4-hidden-layer MLP with perceptron number of 1, 100, 100, 100, 100, 
and 1 for the Online EWC.

Below is the trace of the experiments after each individual task being trained

Task 1:

![loss1_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss1_online4.png?raw=true)
![task1_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task1_online4.png?raw=true)

Task 2:

![loss2_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss2_online4.png?raw=true)
![task2_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task2_online4.png?raw=true)

Task 3:

![loss3_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss3_online4.png?raw=true)
![task3_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task3_online4.png?raw=true)

Task 4:

![loss4_task4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss4_online4.png?raw=true)
![task4_online4](https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/task4_online4.png?raw=true)

Not bad, right? But can we do better? Obviously, Online EWC is not the end, the next section will focus on possible improvements for EWC techniques.

