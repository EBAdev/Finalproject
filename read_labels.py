import gzip


def read_labels(filename: str):
    with gzip.open(filename, 'rb') as f:
        magic_num = int.from_bytes(f.read(4), byteorder="big")
        assert magic_num == 2049, "The magic number of the read file is not 2049"
        return [l for l in f.read()][4:]


if __name__ == "__main__":
    print(read_labels("train-labels-idx1-ubyte.gz"))
