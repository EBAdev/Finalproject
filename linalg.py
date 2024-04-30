"""
Homemade linear algebra module to use for MNIST
"""

from typing import Type

Mat = Type["Matrix"]


class LinAlg:
    def col_space(self):
        """
        Method to return the column space of a LinAlg class
        """
        if isinstance(self.elements[0], (int, float)):
            return 1
        return len(self.elements[0])

    def row_space(self):
        """
        Method to return the row space of a LinAlg class
        """
        if isinstance(self.elements[0], (int, float)):
            return 1
        return len([row for row in self.elements])

    def __iter__(self):
        """
        Method to run when a iterator is called on a LinAlg class
        """
        self.idx = 0
        return self

    def __next__(self):
        """
        Method to return the next element in a LinAlg class
        """
        if self.idx+1 <= len(self.elements):
            x = self.elements[self.idx]
            self.idx += 1
            return x
        else:
            raise StopIteration

    def __getitem__(self, i: int):
        """
        Method to return a element at index of vector
        """
        return self.elements[i]


class Matrix(LinAlg):
    def __init__(self, elements) -> None:
        """
        initiate a 2d-matrix class
        """
        # Input check for matrix
        assert isinstance(elements, list), "elements must be a list"
        assert all(isinstance(sublist, list) for sublist in elements) and all(len(sublist) == len(
            elements[0]) for sublist in elements), "elements must be a list of lists with same length"
        assert all(isinstance(item, (int, float))
                   for sublist in elements for item in sublist), "sublist must contain only integers or floats"

        self.elements = elements
        self.row_vector = False

        if self.row_space() == 1:
            self.row_vector = True

        return None

    def add(self, matrix: Mat) -> Mat:
        assert isinstance(
            matrix, Matrix), "Addition is only defined between two matricies."
        assert self.row_space() == matrix.row_space() and self.col_space() == matrix.col_space(
        ), "addition is only defined between matricies with the same row and column dimension."
        return Matrix([[x+y for (x, y) in zip(row_self, row_other)] for row_self,
                       row_other in zip(self.elements, matrix.elements)])

    def __add__(self, matrix: Mat) -> Mat:
        return self.add(matrix)

    def __str__(self) -> str:
        return f"Matrix: <{self.elements}>"

    def sub(self, matrix: Mat) -> Mat:
        """
        Define subtraction between matricies as the elementwise invese addition
        """
        return self.add(-1*matrix)

    def __sub__(self, matrix: Mat) -> Mat:
        return self.sub(matrix)

    def fact_mult(self, factor: int | float) -> Mat:
        assert isinstance(factor, (int, float)), "factor multiplication of matricies is only defined with integers or floats."
        return Matrix([[factor*x for x in row] for row in self.elements])

    def transpose(self):
        return Matrix([[row[i] for row in self.elements] for i in range(len(self.elements[0]))])

    def mat_mult(self, matrix: Mat) -> Mat:
        assert isinstance(
            matrix, Matrix), "matrix multiplication is only defined between matricies"
        assert self.col_space() == matrix.row_space(
        ), "columnspace and rowspace of the matricies do not match."
        return Matrix([[sum(a * b for a, b in zip(row, col)) for col in zip(*matrix.elements)] for row in self.elements])

    def __mul__(self, other: Mat | int | float) -> Mat | int | float:
        """
        Define multiplication operator to use matrix-product for matricies and scalar multiplication for factors.
        """
        assert isinstance(
            other, (Matrix, int, float)), "Matrix multiplication is only defined with scalars and other matricies"
        if isinstance(other, Matrix):
            return self.mat_mult(other)
        return self.fact_mult(other)

    def __rmul__(self, factor: int | float) -> Mat:
        """
        Define scalarmultiplication for rhs
        """
        assert isinstance(
            factor, (int, float)), "Matrix multiplication is only defined with scalars and other matricies"
        return self.fact_mult(factor)

    def pow(self, n: int) -> str:
        """
        Define powers of vectors as the elementwise power.
        """
        assert isinstance(n, int), "elementwise power of matricies is only defined for factors"
        return Matrix([[x**n for x in row] for row in self.elements])

    def __pow__(self, n: int):
        return self.pow(n)

    def flatten(self) -> Mat:
        return Matrix([[val for row in self.elements for val in row]])

    def reshape(self, cols: int) -> Mat:
        values = self.flatten()[0]
        return Matrix([values[i:i+cols] for i in range(0, len(values), cols)])


if __name__ == "__main__":
    m1 = Matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    m3 = Matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    m2 = Matrix([[1], [2], [3], [4]])

    # print(m1*m2, m1.transpose(), m2.transpose())
    print(m2.reshape(2), m2.flatten())
