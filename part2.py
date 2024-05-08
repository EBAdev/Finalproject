import json

from linalg import Matrix
NetW = list[list[int | float], list[int, float]]
img = list[list[int]]


def linear_load(filename) -> NetW:
    with open(filename) as f:
        weights = json.load(f)
    return weights


def linear_save(filename: str, network: NetW):
    try: # inspiration from: https://www.geeksforgeeks.org/create-a-file-if-not-exists-in-python/
        with open(filename, 'x') as f:
            f.write(str(network))
    except FileExistsError:
        with open(filename, "w") as f:
            f.write(str(network))


def image_to_vector(image: img) -> Matrix:
    return Matrix([[x/255 for row in image for x in row]])


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


def catagorical(label:int, classes=10) -> Matrix:
    """
    Define catagorical 
    """
    assert label <= classes, "labels cannot be longer than classes."
    return Matrix([[1 if i == label else 0 for i in range(classes)]])


def predict(network: NetW, image: img) -> Matrix:
    x = image_to_vector(image)
    A = Matrix(network[0])
    b = Matrix([network[1]])
    return x*A+b


def evaluate(network: NetW, images: list[img], labels: list[int]):
    guesses = [predict(network, img) for img in images]
    predictions = [argmax(guess) for guess in guesses]

    cost = sum([mean_square_error(guesses[i], catagorical(labels[i]))
                for i in range(len(images))])/len(images)

    accuracy = sum([1 if predictions[i] == labels[i]
                   else 0 for i in range(len(images))])/len(images)

    return (predictions, cost, accuracy)


if __name__ == "__main__":
    filename = "mnist_linear.weights"
    nw = linear_load(filename)
    # linear_save(filename, nw)

    from part1 import read_images, read_labels, plot_images
    images = read_images("t10k-images-idx3-ubyte.gz")
    labels = read_labels("t10k-labels-idx1-ubyte.gz")
    # print(labels)
    # print(image_to_vector(images[0]))
    # print(mean_square_error(Matrix([[1, 2, 3, 4]]), Matrix([[3, 1, 3, 2]])))
    # print(argmax(Matrix([[6, 2, 7, 10, 5]])))
    # print(predict(nw, images[0]))
    # print(evaluate(nw, images, labels))
    predicions = evaluate(nw, images, labels)[0]
    plot_images(images, labels, Matrix(nw[0]), predicions)
