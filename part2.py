import json
from linalg import Vector, Matrix
NetW = list[list[int | float], list[int, float]]
img = list[list[int]]


def linear_load(filename) -> NetW:
    with open(filename) as f:
        weights = json.load(f)
    return weights


def linear_save(filename: str, network: NetW):
    try:
        with open(filename, 'x') as f:
            f.write(str(network))
    except FileExistsError:
        with open(filename, "w") as f:
            f.write(str(network))


def image_to_vector(image: img) -> list[int | float]:
    return [x/255 for row in image for x in row]


def mean_square_error(v1: Vector, v2: Vector) -> int | float:
    """
    Define the mean squared error between two vectors
    """
    return sum((v1-v2)**2)/len(v1.elements)


def argmax(v1: Vector) -> int:
    """
    Define argmax for a vector.
    """
    return v1.elements.index(max(v1))


def catagorical(label, classes=10):
    """
    Define catagorical
    """
    assert label <= classes, "labels cannot be longer than classes."
    return [1 if i == label else 0 for i in range(classes)]


def predict(network: NetW, image: img) -> list[Matrix, Vector]:
    x = Matrix([image_to_vector(image)])
    A = Matrix(network[0])
    b = Matrix([network[1]])
    return Vector((x*A+b)[0])


if __name__ == "__main__":
    filename = "mnist_linear.weights"
    nw = linear_load(filename)
    # linear_save(filename, nw)

    from part1 import read_images
    images = read_images("t10k-images-idx3-ubyte.gz")
    # print(image_to_vector(images[0]))
    # print(mean_square_error(Vector([1, 2, 3, 4]), Vector([3, 1, 3, 2])))
    # print(argmax(Vector([6, 2, 7, 10, 5])))
    print(predict(nw, images[0]))
