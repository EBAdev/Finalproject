from linalg import Matrix
from random import shuffle, uniform
from part2 import catagorical, linear_load, image_to_vector

def create_batches(values, batch_size: int):
    #permute the list:
    shuffle(values)
    
    #calculate number of batches
    num_batches = len(values)//batch_size
    
    #indices left out:
    last_size = len(values) % batch_size
    
    #cut into bathes
    batch_list = [values[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)] + [values[-last_size :]]

    return batch_list

def update(network, images, labels):
    A, b = network
    A = Matrix(A)
    b = Matrix(b)
    n = len(images)
    num_classes = b.col_space()
    sigma = 0.1
    for i in range(n):
        x = image_to_vector(images[i])
        a =  x * A + b
        y = catagorical(labels[0])
        error = 1 / 5 * (a - y)
        b -= sigma/n * error
        A -= sigma/n * (x.transpose() * error)
    return A,b



if __name__ == "__main__":
    filename = "mnist_linear.weights"
    nw = linear_load(filename)
    from part1 import read_images, read_labels, plot_images
    images = read_images("t10k-images-idx3-ubyte.gz")
    labels = read_labels("t10k-labels-idx1-ubyte.gz")
    A, b = update(nw, images, labels)
    print(uniform(0,1))
    #print(create_batches(list(range(7)), 3))