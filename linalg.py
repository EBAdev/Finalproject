"""
Homemade linear algebra module to use for MNIST
"""

from typing import Type


Vec = Type["Vector"]
Mat = Type["Matrix"]


class LinAlg:
    def col_space(self):
        if isinstance(self.elements[0], (int, float)):
            return 1
        return len(self.elements[0])

    def row_space(self):
        if isinstance(self.elements[0], (int, float)):
            return 1
        return len([row for row in self.elements])


class Vector(LinAlg):
    def __init__(self, elements: list[int]) -> None:
        """
        Initiate a linear algebra (row)vector.

        Args:
        1. elements (list[int]): A 1-dimensional list of the elements in the vector.
        """
        # Check input
        assert isinstance(elements, list), "elements must be a list"
        assert all(isinstance(item, (int, float))
                   for item in elements), "lists must contain only integers or floats"
        self.elements = elements

    def add(self, vector: Vec) -> Vec:
        """
        Define addition between vectors as the elementwise addition of vectors
        """
        assert isinstance(
            vector, Vector), "Vector addition is only defined between two vectors"
        assert self.col_space() == vector.col_space(
        ), "The column space of the two vectors do not match"

        return Vector([x+y for (x, y) in zip(self.elements, vector.elements)])

    def __add__(self, vector: Vec) -> Vec:
        return self.add(vector)

    def sub(self, vector: Vec) -> Vec:
        """
        Define subtraction between vectors as the elementwise invese addition of vectors
        """
        return self.add(-1*vector)

    def __sub__(self, vector: Vec) -> Vec:
        return self.sub(vector)

    def mult(self, factor: int | float) -> Vec:
        """
        Define multiplication between vectors and scalars as the elementwise scalation
        """
        assert isinstance(
            factor, (int, float)), "factor multiplication of vectors is only defined with integers and floats"
        return Vector([factor * x for x in self.elements])

    def dot(self, vector: Vec) -> int | float:
        """
        Define the dotproduct between two vectors as the classic inner product
        """
        assert isinstance(
            vector, Vector), "Vector addition is only defined between two vectors"
        assert self.col_space() == vector.col_space(
        ), "The column space of the two vectors do not match"
        return sum([x*y for (x, y) in zip(self.elements, vector.elements)])

    def __mul__(self, other: Vec | int | float) -> Vec | int | float:
        """
        Define multiplication operator to use dot-product for vectors and scalar multiplication for factors.
        """
        assert isinstance(
            other, (Vector, int, float)), "Vector multiplication is only defined with scalars and other vectors"
        if isinstance(other, Vector):
            return self.dot(other)
        return self.mult(other)

    def __rmul__(self, factor: int | float) -> Vec:
        """
        Define scalarmultiplication for rhs
        """
        assert isinstance(
            factor, (int, float)), "Vector multiplication is only defined with scalars and other vectors"
        return self.mult(factor)

    def __str__(self) -> str:
        return f"Vector: <{self.elements}>"


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

    def add(self, matrix: Mat) -> Mat:
        assert isinstance(
            matrix, Matrix), "Addition is only defined between two matricies."
        assert self.row_space() == matrix.row_space() and self.col_space() == matrix.col_space(
        ), "addition is only defined between matricies with the same row and column dimension."
        return Matrix([[x+y for (x, y) in zip(row_self, row_other)] for row_self, row_other in zip(self.elements, matrix.elements)])

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
        assert isinstance(
            factor, (int, float)), "factor multiplication of matricies is only defined with integers or floats."
        return Matrix([[factor*x for x in row] for row in self.elements])

    def transpose(self):
        return Matrix([[row[i] for row in self.elements]
                      for i in range(self.row_space())])

    def mat_mult(self, matrix: Mat) -> Mat:
        assert isinstance(
            matrix, Matrix), "matrix multiplication is only defined between matricies"
        assert self.col_space() == matrix.row_space(), "columnspace and rowspace of the matricies do not match."
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

    def __rmul__(self, factor: int | float) -> Vec:
        """
        Define scalarmultiplication for rhs
        """
        assert isinstance(
            factor, (int, float)), "Matrix multiplication is only defined with scalars and other matricies"
        return self.fact_mult(factor)


if __name__ == "__main__":
    v1 = Vector([1, 2, 3])
    v2 = Vector([1, 2, 3])
    print(v1 + v2, v1*2, v1-v2, v1*v2)

    m1 = Matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    m2 = Matrix([[1], [2], [3]])
    print(m2*m1)
