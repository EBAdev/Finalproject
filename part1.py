import gzip
import matplotlib.pyplot as plt

from linalg import Matrix

img = list[list[int]]  

#relevant comment

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
    weight_images = [Matrix([row]).reshape(28) for row in A_T]

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
                    label = f"Failed: {prediction[idx]},\n Correct: {labels[idx]}."
                    color = "Reds"

            ax.imshow(images[idx], cmap=color, vmin=0, vmax=255)
            ax.set_title(label, fontsize=10)
        else:
            normdata = ax.imshow(weight_images[idx-10].elements,
                      cmap="hot", vmin=-1, vmax=1)
            ax.set_title(idx-10, fontsize=10)

   ###få lige det her til at fungere
    fig.colorbar(normdata, ax=axes.ravel().tolist(), shrink=0.8)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=1)
    #plt.tight_layout()
    plt.show()

    return None


if __name__ == "__main__":
    labels = read_labels("t10k-labels-idx1-ubyte.gz")
    images = read_images("t10k-images-idx3-ubyte.gz")
    for row in images[0]:
        print(row)
