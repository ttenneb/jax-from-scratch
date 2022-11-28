from mnist import get_mnist_dataset

train_images, train_labels, test_images, test_labels, training_generator = get_mnist_dataset()
print(train_images[0])