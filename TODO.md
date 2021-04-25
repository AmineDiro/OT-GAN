# Mini batch energy distance : 
- measuring the distance between the distributions of generated data and training data is provided by optimal transport theory.
- Loss GAN : (1)
- Wasserstein distance in the context of GANs (2) => intractable 
- (3) : restriuct to 1-lipschitz func in the dual formulation => still intractable 
= (3)  solutions are well approximated with critics => similar to GAN loss (1)

- Genevay , Cutri  : Sinkhorn , entropic approx of the primal problem
- Mini batch sinkhorn  :
    - Disadvantage is that the expectation of Equation 5 over mini-batches is no longer a valid metric over probability distributions

- Energy Distance, also called Cramer Distance, (6) => valid metric :  sum invariance, scale sensitivity, and
    unbiased sample gradients , loss GAn distance between data real and generated ,
    we need to retro prop the grad of the loss foir each batch , we need an unbiased estimator => Cramer distance

-  Minibatch energy distance : 
    - Use batch instead of images
    - (7) use general distance : generalized energy distance
    - USe Sinkhorn distance in the generalized energy distance =) > unbiased gradient estimator based on OT (8) :
    this is what makes the resulting mini-batch gradients unbiased and the objective statistically consistent. However

- Cost OT-GAN: 
    - Euclidienne distance : statistically inefficient !  
    - Cost : cosine similarity between latent encoding of images ( real and generated)
    - η ( Critic) updated each n_epoch  to maximize the resulting minibatch energy distance 
    - Choice to update Critic less often than Generator (3x) : 
    - Theorem de Danskin , Ignoring the gradient flow through the matchings M is justified by the envelope theorem : 
       Since M is chosen to minimize Wc, the gradient ofWc with respect to this variable is zero 
    - 

# Implementation 
- Architecture : 
- MNIST test , on RTX2070 16go, batch 64  
- Weight init , data init 
- Batch size => 8000 batch size used
- log domain sikhorn ( logsum exp) 
- Image saved each epoch, 

# Resultats : 
- Image : 
    - Commentaires 
- Inception 

# Conclusion : 
- Speed-up for separable kernels).  cost separanle O(n^1+1/d) 
