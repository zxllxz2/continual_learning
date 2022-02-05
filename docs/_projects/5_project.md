---
layout: post
title: "A Step Further from EWC"
description: Masking
---

Experiments with Masking
=========================

Motivation
----------------

Training multiple tasks with EWC performs ideally if the number of tasks is relatively small. However, as the number of tasks increases, the ability to learn new tasks is severely compromised in order to remember previous features. The rationale behind such a phenomenon is the accumulated regularizations of each previous task in calculating the current loss. Maybe you still have some impression about this fomula?

<p align="center">
  <img width="300" height="65" src="https://github.com/zxllxz2/tempweb/blob/main/docs/assets/images/loss_offline_EWC_eq8.jpg?raw=true">
</p>

As we can see, the sum of regularization terms in the loss function increases along with the number of tasks, exerting a severe penalty on any change of parameters. Hence, EWC alone is somewhat insufficient for continual training with a **large** set of tasks. So then, what can we do to improve it?

Binary Mask
--------------
The crux of solving the frozen-parameter problem is to ameliorate the accumulating restrictions exerted by training each task. One promising solution is applying a **binary mask** on gradients. According to the corresponding values in the FIM, the binary mask sets parameters' importance to zero or one with a threshold. A significance level greater than the threshold leads to a zero (frozen) in the mask. In such a manner, the gradients of crucial parameters would become zero with others untouched after multiplying the mask with gradients of the model's parameters element-wisely. Consequently, those crucial parameters of previous tasks would not be updated, while unimportant ones would be optimized during the training.

<img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/mask_pic.jpg?raw=true" style="display:block;margin-left: auto;margin-right: auto;width:78%" />

Layer-wise Binary Mask
-------------------------
For fully connected neural networks, gradients of closer-to-output layers tend to be larger than those near the input layer. So is the FIM. Let's check the significance values of neurons in different layers of our model.

<div style="display: flex;">
  <div style="float: left;width: 60%;padding: 5px;">
    <img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/num_neurons_1l.jpg?raw=true" style="display:flex" />
  </div>
  <div style="float: left;width: 60%;padding: 5px;">
    <img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/num_neurons_lastl.jpg?raw=true" style="display:flex" />
  </div>
</div>

The picture on the left depicts the distribution of neurons' log-scale significance value in the first hidden layer, while the one on the right shows the distribution in the last hidden layer. Obviously, we can see the huge difference in the significance values among different layers. The most frequent significance value in the last hidden layer is about 100000 times larger than those in the first hidden layer in our model.

This observation raises the problem that the neural network would consider layers near the input more significant than those closer to the output. The solution to such a problem is obtaining the binary mask layer-wisely. By freezing an equal portion of parameters for each layer, we can weigh every layer with the same significance.

<img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/mask.jpg?raw=true" style="display:block;margin-left: auto;margin-right: auto; width:60%" />
<br>

We carried out an experiment of layer-wise masking on the MLP we created in the first post, with three hidden layers of a hundred neurons. The mask obtained from the FIM of the first task performs ideally in training the second task when the threshold is set to freeze the most important ninety percent of neurons.

<div style="display: flex;">
  <div style="float: left;width: 70%;padding: 5px;">
    <img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/mask_2tasks.jpg?raw=true" style="display:block;margin-left: auto;margin-right: auto;" />
  </div>
  <div style="float: left;width: 70%;padding: 5px;">
    <img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/mask_2tasks_loss.jpg?raw=true" style="display:flex" />
  </div>
</div>


However, when we apply the mask obtained from the FIM of the second task to train the third task, only when the threshold is set to freeze more than ninety-five percent of neurons can the ANN remember previous tasks. Namely, about five percent of neurons learn the new task but forget learned information. This problem is unexpected.

<div style="display: flex;">
  <div style="float: left;width: 70%;padding: 5px;">
    <img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/mask_3tasks.jpg?raw=true" style="display:block;margin-left: auto;margin-right: auto;" />
  </div>
  <div style="float: left;width: 70%;padding: 5px;">
    <img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/mask_3tasks_loss.jpg?raw=true" style="display:flex" />
  </div>
</div>

Current, one possible solution that we come up with for this problem is to make the neural network sparse. By calculating the L1-norm for each layer, we can get a sparse neural network with most neurons zero. In such a manner, we may only need to manually freeze a few neurons to remember old tasks, while other free neurons would only be capable of learning new tasks without changing previous behaviors. We are currently experimenting this approach.

 That finishes our discussion on the masking. Next we will talk about some possible improvement after applying masking on EWC. Check it if you like this article or are interested in the topic of catastrophic forgetting. Thanks for your support!
