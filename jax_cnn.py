import os
os.add_dll_directory(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin")

# imports
from mnist import get_mnist_dataset
from jax import random, vmap, grad
import jax.numpy as jnp
import jax.nn as jnn


key = random.PRNGKey(22)

class linear_layer:
    def __init__(self, in_size, out_size) -> None:
        self.w  = random.uniform(key, (out_size, in_size), dtype=jnp.float32, minval=-.1, maxval=.1)
        self.b = jnp.zeros(out_size)
        
def relu(x):
    return jnp.maximum(x, 0)

def softmax(x):
    max = jnp.max(jnp.ravel(x))
    x = jnp.exp(x - max)
    x = x / jnp.sum(x, axis=1)
    return x
  
def onehot(x):
    y = jnp.zeros(10)
    y = y.at[x].set(1)
    return y
    
def predict(params, x):
    
    x = jnp.matmul(x, params[0][0].T) + params[0][1]
    x_relu = relu_layer(x)
    x1 = jnp.matmul(x_relu, params[1][0].T) + params[1][1]
    x_softmax = jnn.softmax(x1)
    return x_softmax

# cross entropy loss
def loss(params, x, y):
    y_hat = predict(params, x)
    return jnp.sum(y*(-jnp.log(y_hat)))


train_images, train_labels, test_images, test_labels, training_generator = get_mnist_dataset()

# Define MLP
l1 = linear_layer(784, 128)
l2 = linear_layer(128, 10)
relu_layer = vmap(relu)

# Util functions
onehot_v = vmap(onehot)

# derivative functions
d_loss = grad(loss)


learning_rate = .0005
print([[l1.w, l1.b], [l2.w, l2.b]])
for i, batch in enumerate(training_generator):
    examples, labels = batch
    labels = onehot_v(labels)
    # forward pass
    examples = examples*(1/255)
    gradient = d_loss([[l1.w, l1.b], [l2.w, l2.b]], examples, labels)
    
    
    d_l1, d_l2 = gradient
    l1.w -= learning_rate*d_l1[0]
    l1.b -= learning_rate*d_l1[1]
    l2.w -= learning_rate*d_l2[0]
    l2.b -= learning_rate*d_l2[1]
    
    if i % 5 == 0:
        print('loss', loss([[l1.w, l1.b], [l2.w, l2.b]], examples, labels))

total = 0
for example in zip(test_images, test_labels):
    image, label = example
    image = image.reshape(-1, 784)
    image = image*(1/255)
    y = predict([[l1.w, l1.b], [l2.w, l2.b]], image)
    if jnp.argmax(y) == jnp.argmax(label):
        total += 1
print(total/len(test_images))