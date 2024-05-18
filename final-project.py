import gzip
import random
import matplotlib.pyplot as plt
import json
import csv
from linalg import Matrix

img = list[list[int]]  # 2d object of integer values
NetW = list[list[int | float], list[int, float]]

#part 1
def read_labels(filename: str) -> list[int]:
    """
    Read the labels from a gzip file following the byteroder described in
    http://yann.lecun.com/exdb/mnist/
    Magic number should be 2049

    Args:
    1. filename (str): The filename of the .gz file

    Returns:
    * list[int]: A list of the labels in the file.
    """
    with gzip.open(filename, 'rb') as f:
        magic_num = int.from_bytes(f.read(4), byteorder="big")
        assert magic_num == 2049, "The magic number of the read file is not 2049"
        num_labels = int.from_bytes(f.read(4), byteorder="big")
        return [byte for byte in f.read(num_labels)]


def read_images(filename: str) -> list[img]:
    """
    Read the images from a gzip file following the byteroder described in
    http://yann.lecun.com/exdb/mnist/
    Magic number should be 2051

    Args:
    1. filename (str): The filename of the .gz file

    Returns:
    * list[img]: A list of the images in the file.
    """
    with gzip.open(filename, "rb") as f:
        magic_num = int.from_bytes(f.read(4), byteorder="big")
        assert magic_num == 2051, "The magic number of the read file is not 2051"
        num_img = int.from_bytes(f.read(4), byteorder="big")
        num_row = int.from_bytes(f.read(4), byteorder="big")
        num_col = int.from_bytes(f.read(4), byteorder="big")

        return [[[byte for byte in f.read(num_row)] for _col in range(num_col)] for _img in range(num_img)]


def plot_images(images: list[img], labels: list[int],  Weight_matrix: Matrix, prediction: list[int] = []) -> None:
    """
    Plot the first images in a list of images, along with the corresponding labels.

    Args:
    1. images (list[img]): A list of the images.
    2. labels (list[int]): A list of the image labels.
    3. rows [optional] (int): The amount of image rows to plot.
    4. cols [optional] (int): The amount of image cols to plot.
    5. prediction[optional] (list[int]): A list of predicted labels for the images.

    Returns:
    * Opens a matplotlib plot of the first rows x cols images.
    """

    fig, axes = plt.subplots(nrows= 4, ncols=5, figsize=(10, 8), gridspec_kw={'wspace': 0.5})
    axes1 = axes[0:2]
    axes2 = axes[2:]
    A_T = Weight_matrix.transpose()
    weight_images = [Matrix(row).reshape(28) for row in A_T]

    for idx, ax in enumerate(axes1.flat):
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        # if there is a prediction for image
        color = "gray_r"
        try:
            prediction[idx]
        except IndexError and TypeError:
            label = str(labels[idx])
        else:
            if prediction[idx] == labels[idx]:
                label = f"Correct: {labels[idx]}."
            else:
                label = f"Failed: {prediction[idx]},\n Correct: {labels[idx]}."
                color = "Reds"
        ax.imshow(images[idx], cmap=color, vmin=0, vmax=255)
        ax.set_title(label, fontsize=10)

    for idx, ax in enumerate(axes2.flat):
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        im = ax.imshow(weight_images[idx].elements, cmap="hot", vmin=-1, vmax=1)
        ax.set_title(idx, fontsize=10)
    fig.colorbar(im, ax=axes2[:,-1])
    plt.show()
    return None

#part 2
def linear_load(filename: str) -> NetW:
    """
    Load a json file of filename in as a NetW
    Args:
    1. filename (str): The filename of the .weights file

    Returns:
    * NetW: A network consisting of a list of A and b.
    
    ## Example use
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
    ...     filename = tmp.name
    ...     json.dump([[1, 2], [3, 4]], tmp)
    >>> linear_load(filename)
    [[1, 2], [3, 4]]
    """
    with open(filename) as f:
        weights = json.load(f)
    return weights


def linear_save(filename: str, network: NetW) -> None:
    """
    inspiration from: https://www.geeksforgeeks.org/create-a-file-if-not-exists-in-python/
    Save a .weights file 

    Args:
    1. filename (str): The filename of the .weights file.

    Returns:
    * None: It only saves the .weights file.
    
    ## Example use

    >>> import tempfile
    >>> network = [[1, 2], [3, 4]]
    >>> with tempfile.NamedTemporaryFile(delete=False) as tmp:
    ...     filename = tmp.name
    >>> linear_save(filename, network)
    >>> linear_load(filename)
    [[1, 2], [3, 4]]
    """
    try:  
        with open(filename, 'x') as f:
            f.write(str(network))
    except FileExistsError:
        with open(filename, "w") as f:
            f.write(str(network))
    return None

def image_to_vector(image: img) -> Matrix:
    """
    Takes a image an makes it to a vector and normalize each entry.

    Args:
    1. image (img): an image that satisfies the criteria for the MNIST images.

    Returns:
    * Matrix: a row vector with entries in the range [0,1]
    
    ## Example use
    >>> image = [[0, 255], [127, 255]]
    >>> v1 = image_to_vector(image)
    >>> print(v1)
    |                0.0                1.0 0.4980392156862745                1.0 |
    <BLANKLINE>
    """
    return Matrix([x/255 for row in image for x in row])


def mean_square_error(v1: Matrix, v2: Matrix) -> float:
    """
    Define the mean squared error between two vectors
    
    Args:
    1. v1 (Matrix): The first vector
    2. v2 (Matrix): The second vector

    Returns:
    * float: The mean squared error
    
    ## Example use
    >>> v1 = Matrix([1, 2, 3])
    >>> v2 = Matrix([1, 2, 4])
    >>> mean_square_error(v1, v2)
    0.3333333333333333
    """
    assert v1.row_vector and v2.row_vector, "mean squared error is only defined between row vetors"
    return sum(((v1 - v2)**2)[0])/v1.col_space()


def argmax(v1: Matrix) -> int:
    """
    Define argmax for a vector.
    
    Args:
    1. v1 (Matrix): is a row vector
    
    Returns:
    * int: the index of the largest element of a vector
    
    ## Example use
    >>> v1 = Matrix([1, 2, 3])
    >>> argmax(v1)
    2
    """
    assert v1.row_vector, "argmax is only defined for vectors"
    return v1.elements[0].index(max(v1.elements[0]))


def catagorical(label: int, classes: int = 10) -> Matrix:
    """
    Define catagorical, which is a list where all indeces are 0 besides the number that is given which is 1

    Args:
    1. label (int): a single label
    2. classes (int): the amount of different outcomes

    Returns:
    * Matrix: a row vector (Matrix) of the length classes
    
    # Example use
    >>> print(catagorical(2, 10))
    | 0 0 1 0 0 0 0 0 0 0 |
    <BLANKLINE>
    """
    assert label <= classes, "labels cannot be longer than classes."
    return Matrix([1 if i == label else 0 for i in range(classes)])


def predict(network: NetW, image: img) -> Matrix:
    """
    Returns x * A + b

    Args:
    1. Network (NetW): A network that contain both A and b
    2. Image (img): a single image is given

    Returns:
    * x * A + b
    
    ## Example use
    >>> network = [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], [0.1, 0.2]]
    >>> image = [[0, 255], [127, 255]]
    >>> prediction = predict(network, image)
    >>> print(prediction)
    | 1.3490196078431373 1.6988235294117648 |
    <BLANKLINE>
    """
    x = image_to_vector(image)
    A = Matrix(network[0])
    b = Matrix(network[1])
    return x * A + b


def evaluate(network: NetW, images: list[img], labels: list[int]) -> tuple:
    """
    Evaluates predictions of the numbers, and returns the predictions, accracy of the predictions and the cost.

    Args:
    1. Network (NetW): A network that contain both A and b
    2. images (list[img]): A list of the images.
    3. labels (list[int]): A list of the image labels.
    
    Returns:
    * Predictions (list): is a list of the predictions for the given image
    * cost (float): the value of cost, which is the average MSE
    * Accuracy (float): is the fraction of times we predicted correctly
    """
    guesses = [predict(network, img) for img in images]
    predictions = [argmax(guess) for guess in guesses]

    cost = sum([mean_square_error(guesses[i], catagorical(labels[i]))
                for i in range(len(images))])/len(images)

    accuracy = sum([1 if predictions[i] == labels[i]
                   else 0 for i in range(len(images))])/len(images)

    return (predictions, cost, accuracy)


# part 3
def create_batches(values: list[int | float], batch_size: int) -> list[list[int | float]]:
    """
    Creates permuted batches e.g. 

    Args:
    Values: this is the list that should be made into batches

    Returns:
    * A list of the batches
    """
    random.shuffle(values)

    # https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    return [values[i:i + batch_size] for i in range(0, len(values), batch_size)]


def update(network: NetW, images: list[img], labels: list[int], step_size: float=0.1) -> tuple:
    """
    Updates the network using gradient descent

    Args:
    1. Network (NetW): A network that contain both A and b
    2. images (list[img]): A list of the images.
    3. labels (list[int]): A list of the image labels.
    4. Stepsize (float): a stepsize for the gradient decent

    Returns
    * Tuple containing the elements of A and b that have been updated
    """
    A, b = network

    A = Matrix(A)
    b = Matrix(b)
    n = len(images)

    for img, lab in zip(images, labels):
        x = image_to_vector(img)
        a = x * A + b
        y = catagorical(lab)
        error = 1 / 5 * (a - y)
        b -= step_size/n * error
        A -= step_size/n * (x.transpose() * error)
    return A.elements, b.elements


def learn(images: list[img], labels: list[int], epochs: int, batch_size: int, step_size: float=0.1, test_image_file: str="t10k-images-idx3-ubyte.gz", test_labels_file: str="t10k-labels-idx1-ubyte.gz") -> tuple:
    """
    This function does some training on the data, such that we better can predict the numbers

    Args:
    1. images (list[img]): The list of images 
    2. labels (list[int]): The list of labels
    3. epochs (int): The number of iterations
    4. batch_size (int): The size of the batches
    5. step_size (float): The step size for the gradient descent 
    6. test_image_file (str): The filename for the test images
    7. test_labels_file (str): The filename for the labels that fit with the images

    Returns:
    * Predictions (list): is a list of the predictions for the given image
    * cost (float): the value of cost, which is the average MSE
    * Accuracy (float): is the fraction of times we predicted correctly
    * it also creates a plot of the development of the cost and accurace through the iterations
    """
    test_img = read_images(test_image_file)
    test_labs = read_labels(test_labels_file)

    A_random = [[random.uniform(0, 1/784) for j in range(10)]
                for i in range(784)]
    b_random = [random.random() for i in range(10)]

    print("Random weights generated. Testing")

    linear_save("trained.weights", [A_random, b_random])
    
    evaluation = evaluate([A_random, b_random], test_img, test_labs)
    cost_list = [evaluation[1]] #track cost
    accuracy_list = [evaluation[2]] #track accuracy
    print(f"Test done, cost {evaluation[1]}, accuracy {evaluation[2]}")

    for epoch in range(epochs):
        batch_mask = create_batches(
            [i for i in range(len(images))], batch_size)

        print(f"Itreration --- {epoch} --- ")

        NW = linear_load("trained.weights")

        for idx, batch in enumerate(batch_mask):
            print(f"Batch: {idx} ")
            image_batch = [img for i, img in enumerate(images) if i in batch]
            label_batch = [lab for j, lab in enumerate(labels) if j in batch]

            NW = update(NW, image_batch, label_batch, step_size)


        evaluation = evaluate(NW, test_img, test_labs)
        cost_list.append(evaluation[1])
        accuracy_list.append(evaluation[2])
        
        linear_save("trained.weights", list(NW))


        print(f"Training done, cost: {evaluation[1]}, accuracy {evaluation[2]}")
    

    return evaluation, cost_list, accuracy_list

def plot_ca(cost_list:list, accuracy_list: list) -> None:
    #plot the cost and accuracy
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(accuracy_list, color='blue', marker='', linestyle='-')
    ax2.plot(cost_list, color='blue', marker='', linestyle='-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Time')
    ax1.axhline(y=accuracy_list[-1], color='r', linestyle='--', label = str(accuracy_list[-1]))
    ax1.text(len(accuracy_list) // 2, accuracy_list[-1] - 0.1, str(accuracy_list[-1]), color='r', va='center', ha='right', backgroundcolor='white')
    ax1.set_xticklabels([])
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost over Time')
    ax2.set_xticklabels([])
    
    plt.tight_layout()
    plt.show()

    return None

if __name__ == "__main__":
    #nw = linear_load("mnist_linear.weights")
    #imgs = read_images("train-images-idx3-ubyte.gz")
    #labs = read_labels("train-labels-idx1-ubyte.gz")

    # test_img = read_images("t10k-images-idx3-ubyte.gz")
    # test_labs = read_labels("t10k-labels-idx1-ubyte.gz")

    # Code to learn a new network of random weights.
    #learned = learn(imgs, labs, 5, 100)
    #cost_list = learned[1]
    #accuracy_list = learned[2]
    
    with open('accuracy_list.csv', 'r', newline='') as infile:
        for row in csv.reader(infile):
            acc = row
    with open('cost_list.csv', 'r', newline='') as infile:
        for row in csv.reader(infile):
            cos = row
    cos = [float(cosel) for cosel in cos]
    acc = [float(accel) for accel in acc]
    plot_ca(cos, acc)
    #plot_ca(cost_list, accuracy_list)
    # Code to test trained weight
    # print(evaluate(linear_load("trained.weights"), test_img, test_labs))

    # Code to test random weights
    # print(evaluate(linear_load("random.weights"), read_images(
    #    "train-images-idx3-ubyte.gz"), read_labels("train-labels-idx1-ubyte.gz")))

    #labels = read_labels("t10k-labels-idx1-ubyte.gz")
    #images = read_images("t10k-images-idx3-ubyte.gz")
    #filename = "mnist_linear.weights"
    #nw = linear_load(filename)
    #predicions = evaluate(nw, images, labels)
    #print(f"cost: {predicions[1]} and accuracy: {predicions[2]}")
    #plot_images(images, labels, Matrix(nw[0]), predicions[0])
