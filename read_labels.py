
import gzip


def read_labels(filename: str):
    with gzip.open(filename, 'rb') as f:
        file_content = f.read()
        print(file_content)


if __name__ == "__main__":
    read_labels("t10k-labels-idx1-ubyte.gz")
