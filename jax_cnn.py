from mnist import get_mnist_dataset
from jax import random, vmap, grad
import jax.numpy as jnp
import lovely_jax as lj
lj.monkey_patch()

key = random.PRNGKey(0)

class linear_layer:
    def __init__(self, in_size, out_size) -> None:
        self.w  = random.normal(key, (out_size, in_size), dtype=jnp.float32)
        self.b = jnp.zeros(out_size)
        
def relu(x):
    return jnp.maximum(x, 0)

def softmax(x):
    x = x/(jnp.max(x)/10)
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=0)
  
def onehot(x):
    y = jnp.zeros(10)
    y = y.at[x].set(1)
    return y
    
def predict(params, x):
    
    x = jnp.dot(examples, params[0][0].T) + params[0][1]
    x_relu = relu_layer(x)
    x1 = jnp.dot(x_relu, params[1][0].T) + params[1][1]
    x_softmax = softmax(x1)
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


learning_rate = .0001

for i, batch in enumerate(training_generator):
    examples, labels = batch
    labels = onehot_v(labels)
    
    # forward pass
    
    gradient = d_loss([[l1.w, l1.b], [l2.w, l2.b]], examples, labels)
    
    
    d_l1, d_l2 = gradient
    l1.w -= learning_rate*d_l1[0]
    l1.b -= learning_rate*d_l1[1]
    l2.w -= learning_rate*d_l2[0]
    l2.b -= learning_rate*d_l2[1]
    
    if i % 5 == 0:
        print('loss', loss([[l1.w, l1.b], [l2.w, l2.b]], examples, labels))
    

    