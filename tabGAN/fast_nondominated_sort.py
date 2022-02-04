def fast_non_dominated_sort(func_value_matrix, minimize = False):
    if minimize:
        evaluator = np.less_equal
    else:
        evaluator = np.greater_equal
    nrow = func_value_matrix.shape[0]
    ncol = func_value_matrix.shape[1]
    S = [[] for i in range(nrow)]
    n = np.zeros(nrow)
    #     for i in range(nrow):
    #         for j in range(nrow):
    #             if i != j or i==j:
    #                 comparison = np.sum(evaluator(func_value_matrix[i,], func_value_matrix[j,]))
    #                 if comparison == ncol:
    #                     n[j] += 1
    #                     S[i].append(j)

    for i in range(nrow):
        comparison = np.sum(evaluator(func_value_matrix[i,], func_value_matrix[:,]), axis=1)
        comparison[i] = 0
        indices_dominated = np.where(np.isclose(comparison, ncol))
        n[indices_dominated] += 1
        S[i] = indices_dominated

    F = []
    while True:
        Q = np.where(n == 0)[0]
        n[Q] -= 1
        #         for i in range(nrow):
        #             if n[i] == 0:
        #                 n[i] -= 1
        #                 Q.append(i)
        if len(Q) == 0:
            break
        for i in Q:
            for j in S[i]:
                n[j] -= 1
        F.append(Q)
    return F