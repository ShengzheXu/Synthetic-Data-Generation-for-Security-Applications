"""Distribution utility functions."""

def laplace_smoothing(a_distribution):
    k = 1.0
    tot_k = len(a_distribution) * k
    sum_count = sum(a_distribution)
    p = []
    for component in a_distribution:
        adj_component = (component + k) / (sum_count + tot_k)
        p.append(adj_component)
    print('laplace_smoothing:', a_distribution, p)
    return p