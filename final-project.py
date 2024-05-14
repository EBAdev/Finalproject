import gzip
import random

import matplotlib.pyplot as plt
import json
import random


from linalg import Matrix

img = list[list[int]]  # 2d object of integer values
NetW = list[list[int | float], list[int, float]]


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

        return [[[byte for byte in f.read(num_row)] for col in range(num_col)] for img in range(num_img)]


def plot_images(images: list[img], labels: list[int],  Weight_matrix: Matrix, prediction: list[int] = None,):
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

    fig, axes = plt.subplots(4, 5, figsize=(5, 4))

    A_T = Weight_matrix.transpose()
    weight_images = [Matrix(row).reshape(28) for row in A_T]

    for idx, ax in enumerate(axes.flat):
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        if idx+1 <= 10:
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
                    label = f"Failed: {
                        prediction[idx]},\n Correct: {labels[idx]}."
                    color = "Reds"

            ax.imshow(images[idx], cmap=color, vmin=0, vmax=255)
            ax.set_title(label, fontsize=10)
        else:
            ax.imshow(weight_images[idx-10].elements,
                      cmap="hot", vmin=-1, vmax=1)
            ax.set_title(idx-10, fontsize=10)

    plt.tight_layout()
    plt.show()

    return None


# part 2
def linear_load(filename) -> NetW:
    with open(filename) as f:
        weights = json.load(f)
    return weights


def linear_save(filename: str, network: NetW):
    try:  # inspiration from: https://www.geeksforgeeks.org/create-a-file-if-not-exists-in-python/
        with open(filename, 'x') as f:
            f.write(str(network))
    except FileExistsError:
        with open(filename, "w") as f:
            f.write(str(network))


def image_to_vector(image: img) -> Matrix:
    return Matrix([x/255 for row in image for x in row])


def mean_square_error(v1: Matrix, v2: Matrix) -> Matrix:
    """
    Define the mean squared error between two vectors
    """
    assert v1.row_vector and v2.row_vector, "mean squared error is only defined between row vetors"
    return sum(((v1-v2)**2)[0])/v1.col_space()


def argmax(v1: Matrix) -> int:
    """
    Define argmax for a vector.
    """
    assert v1.row_vector, "argmax is only defined for vectors"
    return v1.elements[0].index(max(v1.elements[0]))  # ! Is this allowed?


def catagorical(label: int, classes=10) -> Matrix:
    """
    Define catagorical
    """
    assert label <= classes, "labels cannot be longer than classes."
    return Matrix([1 if i == label else 0 for i in range(classes)])


def predict(network: NetW, image: img) -> Matrix:
    x = image_to_vector(image)
    A = Matrix(network[0])
    b = Matrix(network[1])
    return x*A+b


def evaluate(network: NetW, images: list[img], labels: list[int]):
    guesses = [predict(network, img) for img in images]
    predictions = [argmax(guess) for guess in guesses]

    cost = sum([mean_square_error(guesses[i], catagorical(labels[i]))
                for i in range(len(images))])/len(images)

    accuracy = sum([1 if predictions[i] == labels[i]
                   else 0 for i in range(len(images))])/len(images)

    return (predictions, cost, accuracy)


# part 3

def create_batches(values: list[int | float], batch_size: int) -> list[list[int | float]]:
    random.shuffle(values)

    # https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    return [values[i:i + batch_size] for i in range(0, len(values), batch_size)]


def update(network, images, labels, step_size=0.1):
    A, b = network

    A = Matrix(A)
    b = Matrix(b)
    n = len(images)

    for img, lab in zip(images, labels):
        x = image_to_vector(img)
        a = x*A + b
        y = catagorical(lab)
        error = 1 / 5 * (a - y)
        b -= step_size/n * error
        A -= step_size/n * (x.transpose() * error)
    return A.elements, b.elements


def learn(images: list[img], labels: list[int], epochs: int, batch_size: int, step_size=0.1, test_image_file="t10k-images-idx3-ubyte.gz", test_labels_file="t10k-labels-idx1-ubyte.gz"):

    test_img = read_images(test_image_file)
    test_labs = read_labels(test_labels_file)

    A_random = [[random.uniform(0, 1/784) for j in range(10)]
                for i in range(784)]
    b_random = [random.random() for i in range(10)]

    print("Random weights generated. Testing")

    linear_save("trained.weights", [A_random, b_random])

    evaluation = evaluate([A_random, b_random], test_img, test_labs)

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

        linear_save("trained.weights", list(NW))
        evaluation = evaluate(NW, test_img, test_labs)
        print(f"Training done, cost: {
              evaluation[1]}, accuracy {evaluation[2]}")

    return evaluation


if __name__ == "__main__":
    nw = linear_load("mnist_linear.weights")
    imgs = read_images("train-images-idx3-ubyte.gz")
    labs = read_labels("train-labels-idx1-ubyte.gz")

    # test_img = read_images("t10k-images-idx3-ubyte.gz")
    # test_labs = read_labels("t10k-labels-idx1-ubyte.gz")

    # Code to learn a new network of random weights.
    print(learn(imgs, labs, 5, 100))

    # Code to test trained weight
    # print(evaluate(linear_load("trained.weights"), test_img, test_labs))

    # Code to test random weights
    # print(evaluate(linear_load("random.weights"), read_images(
    #    "train-images-idx3-ubyte.gz"), read_labels("train-labels-idx1-ubyte.gz")))
