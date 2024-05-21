"""
Homemade linear algebra module to use for MNIST.

This module provides Classes and methods for basic linear algebra

## Classes:
    LinAlg: This is a base class for linear algebra
    Matrix: provides various matrix operations

## Example use
    >>> from linalg import Matrix

    # Create a matrix
    >>> A = Matrix([[1, 2], [3, 4]])

    # Print the matrix
    >>> print(A)
    | 1 2 |
    | 3 4 |
    <BLANKLINE>

    # Matrix addition
    >>> B = Matrix([[5, 6], [7, 8]])
    >>> C = A + B
    >>> print(C)
    |  6  8 |
    | 10 12 |
    <BLANKLINE>
"""

from typing import Type, Union, List


Mat = Type["Matrix"]
matrix_input = Union[List[List[Union[int, float]]], List[Union[int, float]]]


class LinAlg:
    """
    Base class for linear algebra operations
    """

    def col_space(self):
        """
        Method to return the column space of a LinAlg class

        ## Example use
        >>> m = LinAlg()
        >>> m.elements = [[1, 2], [3, 4]]
        >>> m.col_space()
        2
        """
        return len(self.elements[0])

    def row_space(self):
        """
        Method to return the row space of a LinAlg class

        ## Example use
        >>> m = LinAlg()
        >>> m.elements = [[1, 2], [3, 4]]
        >>> m.row_space()
        2
        """
        return len(self.elements)

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
        if self.idx < len(self.elements):
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
    """
    Represents a matrix and provides various matrix operations.
    """

    def __init__(self, elements: matrix_input) -> None:
        """
        initiate a 2d-matrix class

        Args:
        1. Elements of type Union[List[List[Union[int, float]]], List[Union[int, float]]]

        Returns:
        * None

        ## Example use
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.elements
        [[1, 2], [3, 4]]
        >>> v = Matrix([1, 2, 3])
        >>> v.elements
        [[1, 2, 3]]
        """
        assert isinstance(elements, list), "elements must be a list"

        # Vector input
        if all(isinstance(item, (int, float)) for item in elements):
            self.elements = [elements]

        else:  # 2D matrix
            assert all(isinstance(sublist, list) for sublist in elements) and all(len(sublist) == len(
                elements[0]) for sublist in elements), "elements must be a list of lists with same length"
            assert all(isinstance(item, (int, float))
                       for sublist in elements for item in sublist), "sublist must contain only integers or floats"
            self.elements = elements

        self.row_vector = self.row_space() == 1

        return None

    def add(self, matrix: Mat) -> Mat:
        """
        Addition of two matrices of same dimensions

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[5, 6], [7, 8]])
        >>> C = A.add(B)
        >>> print(C)
        |  6  8 |
        | 10 12 |
        <BLANKLINE>
        """
        assert isinstance(
            matrix, Matrix), "Addition is only defined between two matricies."
        assert self.row_space() == matrix.row_space() and self.col_space() == matrix.col_space(
        ), "addition is only defined between matricies with the same row and column dimension."

        return Matrix([[x+y for (x, y) in zip(row_self, row_other)] for row_self,
                       row_other in zip(self.elements, matrix.elements)])

    def __add__(self, matrix: Mat) -> Mat:
        """
        Method for addition of matrices of same dimensions

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[5, 6], [7, 8]])
        >>> C = A + B
        >>> print(C)
        |  6  8 |
        | 10 12 |
        <BLANKLINE>
        """
        return self.add(matrix)

    def __str__(self) -> str:
        """
        Method to print a matrix

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> print(A)
        | 1 2 |
        | 3 4 |
        <BLANKLINE>

        """
        max_width = max(max(len(str(x)) for x in row) for row in self.elements)
        matrix_str = ""
        for row in self.elements:
            matrix_str += "| " + \
                " ".join(f"{x:>{max_width}}" for x in row) + " |\n"
        return matrix_str

    def sub(self, matrix: Mat) -> Mat:
        """
        Define subtraction between matrices as the elementwise inverse addition

        ## Example use
        >>> A = Matrix([[5, 6], [7, 8]])
        >>> B = Matrix([[1, 2], [3, 4]])
        >>> C = A.sub(B)
        >>> print(C)
        | 4 4 |
        | 4 4 |
        <BLANKLINE>
        """
        return self.add(-1 * matrix)

    def __sub__(self, matrix: Mat) -> Mat:
        """
        Subtract Matrices of same dimensions

        ## Example use
        >>> A = Matrix([[5, 6], [7, 8]])
        >>> B = Matrix([[1, 2], [3, 4]])
        >>> C = A - B
        >>> print(C)
        | 4 4 |
        | 4 4 |
        <BLANKLINE>
        """
        return self.sub(matrix)

    def fact_mult(self, factor: int | float) -> Mat:
        """
        Factor multiplication for a matrix and a number

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = A.fact_mult(2)
        >>> print(B)
        | 2 4 |
        | 6 8 |
        <BLANKLINE>
        """
        assert isinstance(
            factor, (int, float)), "factor multiplication of matricies is only defined with integers or floats."
        return Matrix([[factor*x for x in row] for row in self.elements])

    def transpose(self) -> Mat:
        """
        Transpose a matrix

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = A.transpose()
        >>> print(B)
        | 1 3 |
        | 2 4 |
        <BLANKLINE>
        """
        return Matrix([[row[i] for row in self.elements] for i in range(len(self.elements[0]))])

    def mat_mult(self, matrix: Mat) -> Mat:
        """
        Define matrix multiplication for matrices of compatible dimensions

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[2, 0], [1, 2]])
        >>> C = A.mat_mult(B)
        >>> print(C)
        |  4  4 |
        | 10  8 |
        <BLANKLINE>
        """
        assert isinstance(
            matrix, Matrix), "matrix multiplication is only defined between matricies"
        assert self.col_space() == matrix.row_space(
        ), "columnspace and rowspace of the matricies do not match."
        return Matrix([[sum(a * b for a, b in zip(row, col)) for col in zip(*matrix.elements)] for row in self.elements])

    def __mul__(self, other: Mat | int | float) -> Mat | int | float:
        """
        Define multiplication operator to use matrix-product for matricies and scalar multiplication for factors.

        ## Example use
        #matrix multiplication
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[2, 0], [1, 2]])
        >>> C = A * B
        >>> print(C)
        |  4  4 |
        | 10  8 |
        <BLANKLINE>

        #factor multiplication
        >>> D = A * 2
        >>> print(D)
        | 2 4 |
        | 6 8 |
        <BLANKLINE>
        """
        assert isinstance(
            other, (Matrix, int, float)), "Matrix multiplication is only defined with scalars and other matricies"
        if isinstance(other, Matrix):
            return self.mat_mult(other)
        return self.fact_mult(other)

    def __rmul__(self, factor: int | float) -> Mat:
        """
        Define scalarmultiplication for rhs

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = 2 * A
        >>> print(B)
        | 2 4 |
        | 6 8 |
        <BLANKLINE>
        """
        assert isinstance(
            factor, (int, float)), "Matrix multiplication is only defined with scalars and other matricies"
        return self.fact_mult(factor)

    def pow(self, n: int, elementwise: bool = True) -> str:
        """
        Define powers of vectors as the elementwise power.
        

        ## Example use
        #Elementwise
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = A.pow(2)
        >>> print(B)
        |  1  4 |
        |  9 16 |
        <BLANKLINE>

        #Matrix power
        >>> C = A.pow(2, elementwise = False)
        >>> print(C)
        |  7 10 |
        | 15 22 |
        <BLANKLINE>
        """
        assert isinstance(
            n, int), "elementwise power of matricies is only defined for factors"
        if elementwise:
            return Matrix([[x**n for x in row] for row in self.elements])
        else:
            mat = Matrix(self.elements)

            for _ in range(n-1):
                mat *= self
            return mat

    def __pow__(self, n: int):
        """
        Will only work for elementwise power

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = A ** 2
        >>> print(B)
        |  1  4 |
        |  9 16 |
        <BLANKLINE>
        """
        return self.pow(n)

    def flatten(self) -> Mat:
        """
        Flattens the matrix into a vector

        ## Example use
        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = A.flatten()
        >>> print(B)
        | 1 2 3 4 |
        <BLANKLINE>
        """
        return Matrix([[val for row in self.elements for val in row]])

    def reshape(self, cols: int) -> Mat:
        """
        Reshapes the matrix to a square matrix of size cols:

        >>> A = Matrix([1, 2, 3, 4])
        >>> B = A.reshape(2)
        >>> print(B)
        | 1 2 |
        | 3 4 |
        <BLANKLINE>
        """
        values = self.flatten()[0]
        assert (len(values) ** 0.5).is_integer(
        ) or cols == 1, "size of matrix must satsify len((sqrt(matrix))) is an integer"
        return Matrix([values[i:i+cols] for i in range(0, len(values), cols)])



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

