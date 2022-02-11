---
layout: post
title: What's Next?
description: Direction towards Future
bigtop: Potential Improvement
---


A remaining challenge that neither EWC nor masking could address is the failure of training a task that is strongly correlated with old tasks. Weights that are important to new tasks may also be important to strongly correlated old tasks. The failure, thus, may arise as old tasks' important parameters, which should be updated to learn the new task, are frozen to address the forgetting.

The issue leads to a discussion upon the similarity among tasks. Tasks are considered similar if they share approximately the same distribution of significant parameters. Since masks are derived directly from the FIM, similarity in the zero-one pattern of binary masks implies similarity among tasks. Therefore, masking can be a powerful tool for task-similarity comparison.

A proposed solution to address the problem based on the similarity analysis is to use a novel approach called TRGP (Trust Region Gradient Projection), which leverages the knowledge of the strongly correlated old tasks through a scaled weight projection (TRGP: Trust Region Gradient Projection for Continual Learning, in communication, 2022). Particularly, a scaling matrix is learned in each layer for the new task to scale the weight projection onto the subspace of correlated old tasks, in order to reuse the frozen weights of old tasks without modifying the model.

<img src="https://github.com/zxllxz2/continual_learning/blob/main/docs/assets/images/trust_region.jpg?raw=true" style="display:block;margin-left: auto;margin-right: auto;width:84%" /> <br>
Apart from its application in relieving catastrophic forgetting, the layer-wisely measured similarity among different tasks can reveal more interesting aspects of ANNs. For instance, we can draw correspondences between ANN's layers and input features by comparing the similarity among tasks layer-wisely. The benefit of such analysis is that in future training, we can choose to freeze certain layers and only allow layers that capture the varied features of the new task to be updated based on the knowledge of input.

So that's it! What you have just read in those six posts are what we have done during Fall 2021. We walked through some regularization-based methods which demonstrate effective mitigation in the catastrophic forgetting issue, and also explored some mixing techniques, like masking and trust regions (memory-based approach), from different directions.

Our future work may explore how tasks with overlapping data points may affect the regularization-based technique. Moreover, we'll focus on adapting our approach to higher-dimensional spaces, as well as to other problem types besides regression, including classification and clustering.

Again, thank you so much for reading our posts! If you are interested in topics like continual learning and catastrophic forgetting, you are more than welcome to visit <a href="http://lab.vanderbilt.edu/mint-lab" target="_blank">our lab website</a> and stay tuned for our future update. We will continue update this website as we proceed our survey and experiments.

Reference
------
- TRGP: Trust Region Gradient Projection for Continual
  Learning. In Tenth International Conference on Learning
  Representations, ICLR 2022. ICLR, 2022. In communication.

<a href="#top">Back to top</a>

<p style="text-align:center; display: flex; justify-content: space-between">
  <a href="../5_project">Prev: Masking</a>
  <a href="../1_project">Back: Introduction</a>
</p>
