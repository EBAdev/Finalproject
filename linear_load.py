"""
Write functions linear_load(file_name) and linear_save(file_name, network) to load and save a linear classifier network = (A, b) using JSON. Test your functions on mnist_linear.weights.
"""
import json

NetW = list[list[int | float], list[int, float]]


def linear_load(filename) -> NetW:
    with open(filename) as f:
        weights = json.load(f)
    return [weights[0], weights[1]]


def linear_save(filename: str, network: NetW):
    try:
        with open(filename, 'x') as f:
            f.write(str(network))
    except FileExistsError:
        with open(filename, "w") as f:
            f.write(str(network))


if __name__ == "__main__":
    filename = "mnist_linear.weights"
    nw = linear_load(filename)
    linear_save(filename, nw)
