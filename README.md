# attention-cluster

a cool trick using a parameter-free attention algorithm to let us perform soft clustering that is both differentiable and gpu accelerated.


## What is soft clustering?
Typical clustering methods like K-Means at every step will strictly define points to be part of a cluster based on a max score or best choice.
In soft clustering, we instead end up with a set of probabilties for each cluster and define our new value to be a weighted sum using the probabilties as coefficients. In practice, we will use very low temperatures (high scale) to give us very peaky distributions and basically recover the behavior of hard-clustering algorithms. In other words, probability of 1 for one cluster and 0 everywhere else.
These tricks allow us to frame everything as matrix multiplications making it gpu friendly and fully differentiable.

## Use cases
- accelerate clustering using gpus
- differentiable parameter-free segmentation based on similar features
- differentiable parameter-free top-k cluster centers based on similar features
- loss functions that ask a generated image to have certain set of colors.
see notebook for examples
