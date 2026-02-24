import numpy as np

A = np.array([
  [1, 2, 3],
  [3, 1, 2],
  [0, 0, 1]
])

def laplace_expansion_rows_determinant(A, loops = 0):
    n = A.shape[0]
    if n == 1:
        return (A[0, 0], loops)
    i = find_most_zeros_row_index(A)
    determinant = 0
    for k in range(n):
      if A[i, k] == 0:
        continue
      
      submatrix = np.delete(np.delete(A, i, axis=0), k, axis=1)
      sign = (-1) ** (i + k)
      minor, loops = laplace_expansion_rows_determinant(submatrix, loops + 1)
      determinant += sign * A[i,k] * minor
    return (determinant, loops)

def find_most_zeros_row_index(A):
    i = 0
    for j, row in enumerate(A):
        if np.count_nonzero(row) < np.count_nonzero(A[i]):
            i = j
    return i

det_A = np.linalg.det(A)
print("Numpy Determinant:", det_A)
print("Determinant:", laplace_expansion_rows_determinant(A))