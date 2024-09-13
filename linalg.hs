module LinAlg where

import Data.List (find)
import Data.Maybe (isNothing)
import Text.Printf (printf)


main :: IO()
main = do
    let mat = [[1,-3,3],[3,-5,3],[6,-6,4]]
    let test = [[1,0],[0,1],[0,1]]
    let v = [1,2,0]
    let w = [[1,2,0],[8,1,-6],[0,0,1]]
    printMatrix (inverse mat)
    putStrLn "\n"
    print (det mat)
    putStrLn "\n"
    printVector (projection v w)
    putStrLn "\n"
    printMatrix (gramSchmidt w)
    putStrLn "\n"
    printMatrixDecomposition (qrDecomposition mat)
    printVector (eigenvalues mat)
    putStrLn "\n"
    printMatrixDecomposition (svd mat)
    printVector (singularValues mat)
    putStrLn "\n"
    printMatrix (pseudoInverse test)


type Vector = [Double]
type Matrix = [[Double]]

--
-- printing functions
--
printVector :: Vector -> IO()
printVector v = do
  putStr "["
  addSpaceVect v
  where
    decimals = 5  -- number of decimals to show
    format = "%." ++ show decimals ++ "f"
    addSpaceVect x
      | length x == 1 = do
          putStr (printf format (head x) ++ "]\n")
      | otherwise = do
          putStr (printf format (head x) ++ "  ")
          addSpaceVect (tail x)


printMatrix :: Matrix -> IO()
printMatrix [] = return ()
printMatrix (x:xs) = do
    printVector x
    printMatrix xs

printMatrixDecomposition :: [Matrix] -> IO()
printMatrixDecomposition [] = return ()
printMatrixDecomposition (x:xs) = do
  printMatrix x
  putStrLn "\n"
  printMatrixDecomposition xs


--
-- math functions on matrices and vectors
--
numRows :: Matrix -> Int
numRows = length

numCols :: Matrix -> Int
numCols mat = length (head mat)


-- returns a zero vector of length 'n'
zeroVector :: Int -> Vector
zeroVector n = replicate n 0


-- returns the input vector that has been multiplied by 'n'
vectorScalarProduct :: Double -> Vector -> Vector
vectorScalarProduct n = map (* n)


-- returns the input matrix that has been multiplied by 'n'
matrixScalarProduct :: Double -> Matrix -> Matrix
matrixScalarProduct n = map (vectorScalarProduct n)


-- returns the sum of two vectors
vectorSum :: Vector -> Vector -> Vector
vectorSum = zipWith (+)


-- returns the dot product between two vectors
dotProduct :: Vector -> Vector -> Double
dotProduct v w = sum (zipWith (*) v w)


-- return the norm of a vector
vectorNorm :: Vector -> Double
vectorNorm v = sqrt (dotProduct v v)


-- returns the input vector divided by its norm to make it unitary
unitVector :: Vector -> Vector
unitVector v = vectorScalarProduct (1 / vectorNorm v) v


-- returns the sum of two matrices
matrixSum :: Matrix -> Matrix -> Matrix
matrixSum = zipWith vectorSum


-- returns the transpose of a matrix
transpose :: Matrix -> Matrix
transpose ([]:_) = []
transpose mat = map head mat : transpose (map tail mat)


-- returns the product of two matrices
matrixProduct :: Matrix -> Matrix -> Matrix
matrixProduct mat1 mat2 = [map (dotProduct row) (transpose mat2) | row <- mat1]


-- returns a matrix where the second one is appended to the first one (to the right)
concatMatrix :: Matrix -> Matrix -> Matrix
concatMatrix = zipWith (++)


-- transforms a matrix into Row Echelon Form (ref)
ref :: Matrix -> Matrix
ref m = f m 0 [0 .. rows - 1]
  where 
    rows = numRows m
    cols = numCols m

    f m _ [] = m
    f m lead (r : rs)
      | isNothing indices = m
      | otherwise = f m' (lead' + 1) rs
      where 
        indices = find p l
        p (col, row) = m !! row !! col /= 0
        l = [(col, row) | col <- [lead .. cols - 1], row <- [r .. rows - 1]]

        Just (lead', i) = indices   -- position of the current pivot (col, row)
        currRow = m !! i            -- current row
        
        m' = zipWith g [0..] m  -- for all rows in m, apply g to eliminate entries in other rows
        g n row
          | n <= r = row   -- if it is a row above the pivot row, keep it the same
          | otherwise = zipWith h currRow row
          where h = subtract . (* (row !! lead' / currRow !! lead'))


-- transforms a matrix into Reduced Row Echelon Form (rref)
rref :: Matrix -> Matrix
rref m = f m 0 [0 .. rows - 1]
  where 
    rows = numRows m
    cols = numCols m

    f m _ [] = m
    f m lead (r : rs)
      | isNothing indices = m
      | otherwise = f m' (lead' + 1) rs
      where 
        indices = find p l
        p (col, row) = m !! row !! col /= 0
        l = [(col, row) | col <- [lead .. cols - 1], row <- [r .. rows - 1]]

        Just (lead', i) = indices
        newRow = map (/ m !! i !! lead') (m !! i)  -- divide whole row by the pivot
        tempM = replace i newRow m                  -- replace the pivot row by the new one

        m' = zipWith g [0..] tempM  -- for all rows in m, apply g to eliminate entries in other rows
        g n row
            | n == r = row   -- if it is the current row, keep it the same
            | otherwise = zipWith h newRow row
            where h = subtract . (* row !! lead')


-- Replaces the n-th element at the given index
replace :: Int -> a -> [a] -> [a]
replace i elem list = a ++ elem : b
  where (a, _ : b) = splitAt i list


-- gets a slice of a list
-- from (inclusive), to (inclusive)
slice :: Int -> Int -> [a] -> [a]
slice from to xs = take (to - from + 1) (drop from xs)


-- returns the slice of a matrix with the specified indices
-- fromRow (inclusive), toRow (inclusive)
-- fromCol (inclusive), toCol (inclusive)
matrixSlice :: Int -> Int -> Int -> Int -> Matrix -> Matrix
matrixSlice fromR toR fromC toC mat = [slice fromC toC row | row <- slice fromR toR mat]


-- returns the diagonal of a matrix
diagonal :: Matrix -> Vector
diagonal [] = []
diagonal mat = head (head mat) : diagonal (map tail (tail mat))


-- returns an n-by-n identity matrix
identityMatrix :: Int -> Matrix
identityMatrix n = [replace i 1 (zeroVector n) | i <- [0 .. n - 1]]


-- appends the identity matrix at the end of a matrix (the right)
addIdentity :: Matrix -> Matrix
addIdentity mat = concatMatrix mat (identityMatrix (numRows mat))


-- return the inverse of a matrix
inverse :: Matrix -> Matrix
inverse mat
  | rows /= cols = mat
  | otherwise = map (slice cols (2 * cols - 1)) (rref (addIdentity mat))
  where
    rows = numRows mat
    cols = numCols mat


-- returns the determinant of a matrix and 0 if it is not invertible
det :: Matrix -> Double
det [] = 1
det x
  | rows /= cols = 0  -- if matrix is not square, return 0 as it is not invertible
  | otherwise = product (diagonal mat)
  where
    mat = ref x
    rows = numRows x
    cols = numCols x


-- returns the vector corresponding to the projection of v on w
-- the rows of w represent a set of vectors
projection :: Vector -> Matrix -> Vector
projection v w = sumVectors [proj v vect | vect <- w]
  where
    proj a b = vectorScalarProduct (dotProduct a b / dotProduct b b) b  -- projection de Vect a sur Vect b
    sumVectors (x:xs) = foldr vectorSum x xs  -- adds all vectors together


-- takes a matrix where the rows correspond to vectors
-- returns a matrix where the rows (vectors) have been orthogonalized by Gram-Schmidt process
gramSchmidt :: Matrix -> Matrix
gramSchmidt [] = []
gramSchmidt (x:xs) = x : gramSchmidt [zipWith subtract (proj row x) row | row <- xs]
  where
    proj a b = vectorScalarProduct (dotProduct a b / dotProduct b b) b


-- returns the QR decompositon of a matrix
-- first element is Q and second is R
qrDecomposition :: Matrix -> [Matrix]
qrDecomposition mat = [q, matrixProduct (transpose q) mat] 
  where 
    qMatrix x = transpose [unitVector row | row <- gramSchmidt (transpose x)]
    q = qMatrix mat


-- returns a vector of the eigenvalues of a matrix
-- first input is the number of iterations to accomplish
eigenvalues :: Matrix -> Vector
eigenvalues = eigenvaluesTemp 100
  where
    -- auxiliary function of eigenvalues
    -- iter is the number of iterations we run the algorithm for
    eigenvaluesTemp iter mat
      | iter == 0 = diagonal (matrixProduct (last qr) (head qr))
      | otherwise = eigenvaluesTemp (iter - 1) (matrixProduct (last qr) (head qr))
      where
        qr = qrDecomposition mat


-- returns the reduced SVD decomposition of a matrix
-- svd = [u, s, transpose v]
-- S matrix is always square
svd :: Matrix -> [Matrix]
svd x = svdTemp 100 [identityMatrix (numRows x), x, identityMatrix (numCols x)]
  where 
    -- auxiliary function of svd
    -- iter is the number of iterations we run the algorithm for
    svdTemp iter mat
      | iter == 0 = [head mat, mat !! 1, transpose (last mat)]
      | otherwise = svdTemp (iter - 1) [matrixProduct (head mat) (head us), transpose (last sv), matrixProduct (last mat) (head sv)]
      where
        us = qrDecomposition (mat !! 1)
        sv = qrDecomposition (transpose (last us))


-- returns the singular values of a matrix
singularValues :: Matrix -> Vector
singularValues mat = diagonal (svd mat !! 1)


-- returns the Moore-Penrose pseudo-inverse of a matrix
pseudoInverse :: Matrix -> Matrix
pseudoInverse mat = matrixProduct (transpose (last temp)) (matrixProduct (inverse (temp !! 1)) (transpose (head temp)))
  where
    temp = svd mat