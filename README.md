# GAN_v1
My implementation to test my understanding of Generative Adversarial Networks

# requirements
Pytorch, opencv-python, numpy

# Why I picked MNIST?
I picked classical mnist dataset for simplicity because I was not sure if my understanding was correct

# Generate images
The generated images are inside `gen_img` folder (looks good to me, except it has tendency to generate certain numbers). To generate images, run `test_model.py`

# Training model 
run `net.py`, sorry the code is not in a good structure because I was only focusing on implementing the theory I learned.

# Improvements
It is hard to use this model on larger images (I have tested it, it failed. The result is on my GAN_v1_failure repo). I am focusing on learning how to improve the loss functions so the gradient is visible by the generator, also I will check more papers about how this is scaled.
