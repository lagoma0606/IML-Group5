if retention == 'OG':
    pruned_x, pruned_y = [], []
    for i, sample in enumerate(x_train):
        rest_x = [x_train[j] for j in range(len(x_train)) if j != i]
        rest_y = [y_train[j] for j in range(len(y_train)) if j != i]

        if len(rest_x) > 0:
            dists = [calc_dist(sample, r_x) for r_x in rest_x]
            nn_idx = int(np.argmin(dists))
            if rest_y[nn_idx] != y_train[i]:
                pruned_x.append(sample)
                pruned_y.append(y_train[i])

    if len(pruned_x) > 0:
        x_train, y_train = pruned_x, pruned_y