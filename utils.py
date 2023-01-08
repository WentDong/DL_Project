import jittor as jt
from jittor import nn
import scipy.spatial as spa # for Delaunay triangulation
import itertools
from matplotlib import pyplot as plt
def local_response_norm(input, size, alpha, beta, k):
    r"""Applies local response normalization over an input signal composed of 
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.
    Rewrite according to the torch version.

    input: The input signal, jittor.Var, The first dimension is batchsize!
    size: int
    alpha: float
    beta: float
    k:  float
    """    
    dim = input.ndim
    if dim < 3:
        raise ValueError(
            "Expected 3D or higher dimensionality\
                input (got {} dimensions)".format(
                    dim
                )
        )

    if input.numel() == 0:
        # Empty
        return input
    
    div = jt.multiply(input, input) # Element-wise mul
    if dim == 3:
        div = nn.pad(div, (0,0, size//2, (size-1)//2))
        div = nn.avg_pool2d(div, (size,1), 1)
        div = jt.squeeze(div, 1)
    else:
        sizes = input.shape
        div = jt.view(div, sizes[0], 1, sizes[1], sizes[2], -1)
        div = nn.pad(div, (0,0,0,0,size//2, (size -1)//2))
        div = nn.AvgPool3d((size,1,1), 1)(div)
        div = jt.squeeze(div, 1)
        div = jt.view(div, sizes)
    div = div.multiply(alpha).add(k)
    div = jt.pow(div, beta)
    return input/div


def l2norm(node_feat):
    r"""
    Implement for local response norm.
    """
    beta = 0.5
    k = 0
    alpha = node_feat.shape[1] * 2 
    size = node_feat.shape[1] * 2
    return local_response_norm(node_feat, size, alpha, beta, k)


def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = jt.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

def plot_image_with_graph(img, kpt, A=None):
    plt.imshow(img)
    plt.scatter(kpt[0], kpt[1], c='w', edgecolors='k')
    if A is not None:
        for idx in jt.nonzero(A):
            plt.plot((kpt[0, idx[0]], kpt[0, idx[1]]), (kpt[1, idx[0]], kpt[1, idx[1]]), 'k-')
