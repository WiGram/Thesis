S = 2
A = 3


L = np.zeros((S,A,A))

for i in range(A):
    for k in range(i + 1):
        tmp_sum = sum(L[i, j] * L[k, j] for j in range(k))

        if i == k:
            L[i, k] = np.sqrt(A[i, i] - tmp_sum)