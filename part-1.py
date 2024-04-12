import gzip
import matplotlib.pyplot as plt

img = list[list[int]]  # 2d object of integer values


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



def plot_images(images: list[img], labels: list[int], rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    i = 0
    for ax, image, label in zip(axes.flat, images, labels):
        ax.imshow(image, cmap='gray_r', vmin=0, vmax=255)
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax.set_title(label)
    plt.show()


if __name__ == "__main__":
    labels = read_labels("t10k-labels-idx1-ubyte.gz")
    images = read_images("t10k-images-idx3-ubyte.gz")
    plot_images(images, labels)
