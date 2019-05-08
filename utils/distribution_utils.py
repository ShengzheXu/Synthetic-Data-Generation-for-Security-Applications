"""Distribution utility functions."""

def get_distribution_with_laplace_smoothing(a_count):
    k = 1.0
    tot_k = len(a_count) * k
    sum_count = sum(a_count)
    p = []
    for component in a_count:
        adj_component = (component + k) / (sum_count + tot_k)
        p.append(adj_component)
    # print('laplace_smoothing:\n', a_count, '\n', p)
    return p