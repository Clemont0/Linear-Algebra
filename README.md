# Linear-Algebra
Linear algebra package written in Haskell

For this package. a Vector is simply a list of doubles [Double] and a Matrix is a list of Vectors so [[Double]].

## 1 - Printing functions
a)  printVector: prints a vector where entries have a fixed number of decimals which can be set with the 'decimals' variable.  
b)  printMatrix: prints a matrix by printing every row as a vector.  
c)  printMatrixDecomposition: prints a list of matrices with a newline between each to distinguish them.  

## 2 - Linear algebra functions  
a)  numRows & numCols  
- input: a matrix
- return: the number of rows and the number of columns of a matrix, respectively

b)  zeroVector
- input: an integer
- return: a zero vector of the specified length

c)  vectorScalarProduct
- input: a double and a vector
- return: the vector scaled by the double

d)  matrixScalarProduct
- input: a double and a matrix
- return: the matrix scaled by the double

e)  vectorSum
- input: two vectors
- return: the vector resulting from the sum of both vectors

f)  dotProduct
- input: two vectors
- return: the dot product of the two vectors, as a double

g)  vectorNorm
- input: a vector
- return: the norm of the vector, as a double

h)  unitVector
- input: a vector
- return: the vector divided by its norm to make it a unit vector (norm of 1)

i)  matrixSum
- input: two matrices
- return: the matrix resulting from the sum of both matrices

j)  transpose
- input: a matrix
- return: the transpose of the matrix

k)  matrixProduct
- input: two matrices
- return: the matrix resulting from the product of both matrices

l)  concatMatrix
- input: two matrices
- return: the matrix resulting from adding the second matrix to the right of the first one (essentially an augmented matrix)

m)  ref
- input: a matrix
- return: the matrix in row echelon form (upper triangular)

n)  rref
- input: a matrix
- return: the matrix in reduced row echelon form

o)  replace
- input: an integer (i), a new element of some type 'a' and a list of same type 'a'
- return: a list of same type 'a' where i-th element has been replaced by the new element

p)  slice
- input: two integers and a list of some type 'a'
- return: a list same type 'a' of the elements from the first argument (inclusive) to the second argument (inclusive)

q)  matrixSlice
- input: four integers and a matrix
- return: a matrix with the rows from the first argument (inclusive) to the second argument (inclusive) and with the columns from the third argument (inclusive) to the fourth argument (inclusive)

r)  diagonal
- input: a matrix
- return: a vector containing the elements of the main diagonal of the matrix

s)  identityMatrix
- input: an integer (n)
- return: an n-by-n identity matrix

t)  addIdentity
- input: a matrix
- return: a matrix where the identity matrix is added to the right to form an augmented system (a specific usage of the concatMatrix function that comes up often)

u)  inverse
- input: a matrix
- return: the matrix itself if not invertible, otherwise the inverse of the matrix

v)  det
- input: a matrix
- return: 0 if the matrix is not invertible, otherwise the determinant of the matrix

w)  projection
- input: a vector and a matrix
- return: the vector resulting from the projection of the vector on the rows of the matrix (the rows of the matrix represent a set of vectors)

x)  gramSchmidt
- input: a matrix
- return: the matrix resulting from applying the Gram-Schmidt process on the rows of the matrix (orthogonalizing its rows)

y)  qrDecomposition
- input: a matrix
- return: a list of matrices where the first element is the Q matrix and the second element is the R matrix resulting from the QR decomposition of the input matrix

z)  eigenvalues
- input: a matrix
- return: a vector containing the eigenvalues of the matrix

aa)  svd 
- input: a matrix
- return: a list of matrices where the first element is the U matrix, the secone element is the S matrix and the last element is the tranpose of the V matrix resulting from the Singular Value Decomposition of the input matrix

bb)  singularValues
- input: a matrix
- return: a vector containing the singular values of the matrix

cc)  pseudoInverse
- input: a matrix
- return: the Moore-Penrose pseudo-inverse of the matrix
