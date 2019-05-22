import numpy as np


def dctii(v, normalized=True, sampling_factor=None):
    """
    Computes the inverse discrete cosine transform of type II,
    cf. https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

    Args:
        v: Input vector to transform
        normalized: Normalizes the output to make output orthogonal
        sampling_factor: Can be used to "oversample" the input to create overcomplete dictionaries

    Returns:
        Discrete cosine transformed vector
    """
    n = v.shape[0]
    K = sampling_factor if sampling_factor else n
    y = np.array([sum(np.multiply(v, np.cos((0.5 + np.arange(n)) * k * np.pi / K))) for k in range(K)])
    if normalized:
        y[0] = 1 / np.sqrt(2) * y[0]
        y = np.sqrt(2 / n) * y
    return y


def dictionary_from_transform(transform, n, K, normalized=True, inverse=True):

    H = np.zeros((K, n))
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        H[:, i] = transform(v, sampling_factor=K)
    if inverse:
        H = H.T
    return np.kron(H.T, H.T)


def overcomplete_idctii_dictionary(n, K):

    if K > n:
        return dictionary_from_transform(dctii, n, K, inverse=False)
    else:
        raise ValueError("K needs to be larger than n.")


def unitary_idctii_dictionary(n):

    return dictionary_from_transform(dctii, n, n, inverse=False)
