# %% Import

import sympy as spy
import numpy as np
from itertools import permutations

# %% Generate tensors

def t(n: int, s: str = "a", dims: list=None) -> spy.Array:
    """Generate tensor.

    Generate n-th-order tensor based on string `s` for the 
    components and dimensions `dims`.

    Parameters
    ----------
    n : int
        Tensor order
    s : str, optional
        Base string for components, by default "a"
    dims : list, optional
        List of tensor dimensions, by default None.
        * For `dims == None`, `dims` is set to `[3]*n`

    Returns
    -------
    spy.Array
        Tensor
    """
    if dims == None:
        dims = [3]*n
    head = f"{s}:"
    dims_pattern = ":".join([str(d) for d in dims])
    symbol_pattern = head + dims_pattern 
    symbols = spy.symbols(symbol_pattern, real=True)
    t = spy.Array(symbols, shape=dims)
    return t


# %% Transform

def flatten(a: spy.Array) -> spy.Array:
    return a.reshape(spy.prod(a.shape))


def vec(a: spy.Array) -> spy.Matrix:
    return a.reshape(spy.prod(a.shape), 1).tomatrix()


def nvn(a: spy.Array) -> spy.Matrix:
    convention = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0,1]]
    if a.shape == (3, 3):
        out = spy.Matrix([
            a[0, 0],
            a[1, 1],
            a[2, 2],
            a[1, 2]*SR2,
            a[0, 2]*SR2,
            a[0, 1]*SR2
        ])
    else:
        out = spy.eye(6)
        for i_1 in range(6):
            for i_2 in range(6):
                out[i_1, i_2] = a[
                    convention[i_1][0],
                    convention[i_1][1],
                    convention[i_2][0],
                    convention[i_2][1]
                ]
                if i_1 > 2:
                    out[i_1, i_2] *= SR2
                if i_2 > 2:
                    out[i_1, i_2] *= SR2
    return out


def nvn_inv(a: spy.Array) -> spy.Array:
    #TODO
    pass


def inv_nvn(a: spy.Array) -> spy.Array:
    # TODO
    pass


# %% Products

def sp(
    a: spy.Array, 
    b:spy.Array
    ):
    return vec(a).dot(vec(b))


def nf(a: spy.Array):
    return vec(a).norm()


def td(
    a: spy.Array, 
    b: spy.Array, 
    n: int
    ) -> spy.Array:
    if a.shape == b.shape and n == a.rank():
        return sp(a, b)
    else:
        return spy.Array(np.tensordot(a, b, n))


def tp(*args):
    temp = args[0]
    for i in range(1, len(args)):
        temp = spy.tensorproduct(temp, args[i])
    return temp


def tc(a: spy.Array, i: list):
    return spy.tensorcontraction(a, i)


def lm(a: spy.Array, b: spy.Array):
    return td(a, b, b.rank())


def rp(q: spy.Array, a: spy.Array):
    ra = a.rank()
    if ra == 1:
        return spy.Array(q.tomatrix()*spy.Matrix(a))
    else:
        con = (1, ra + 2 - 1)
        temp = tc(tp(q, a), con)
        for i in range(ra - 1):
            temp = tc(tp(q, temp), con)
        return temp


# %% Linear algebra

def linsolve_t(eqs: spy.Array, vs=None) -> list:
    if vs == None:
        vs = eqs.free_symbols
    if eqs.is_scalar:
        eqs = [eqs]
    else:
        eqs = flatten(eqs)
    sol = list(spy.linsolve(eqs, *vs))[0]
    return [(v, s) for v, s in zip(vs, sol)]


def orthogonalize_basis(basis: spy.Array) -> spy.Array:
    n = basis.shape[0]
    if basis.rank() > 2:
        dims = basis.shape[1:]
    gs = spy.GramSchmidt([vec(b) for b in basis], orthonormal=True)
    gs = spy.Array([spy.Array(g) for g in gs])
    if basis.rank() == 2:
        return gs
    else:
        return gs.reshape(n, *dims)


# %% Transpositions

def tt(a: spy.Array, axes) -> spy.Array:
    return spy.permutedims(a, axes)


def tt_m(a: spy.Array) -> spy.Array:
    ra = a.rank()
    front = list(range(ra//2))
    back = list(range(ra//2, ra))
    return tt(a, back + front)


# %% Symmetrizations

def symmetrize(a: spy.Array) -> spy.Array:
    p = list(permutations(list(range(a.rank()))))
    for i in p:
        eqs = a - spy.permutedims(a, i)
        vs = eqs.free_symbols
        if len(vs) > 0:
            a = a.subs(linsolve_t(eqs))
    return a


def symmetrize_explicit(
    a: spy.Array, 
    p: list
    ) -> spy.Array:
    eqs = a - spy.permutedims(a, p)
    vs = eqs.free_symbols
    if len(vs) > 0:
        a = a.subs(linsolve_t(eqs))
    return a


def symmetrize_l(a: spy.Array) -> spy.Array:
    return symmetrize_explicit(a, (1, 0, 2, 3))


def symmetrize_r(a: spy.Array) -> spy.Array:
    return symmetrize_explicit(a, (0, 1, 3, 2))


def symmetrize_m(a: spy.Array) -> spy.Array:
    return symmetrize_explicit(a, (2, 3, 0, 1))


def symmetrize_lr(a: spy.Array) -> spy.Array:
    return symmetrize_r(symmetrize_l(a))


def symmetrize_lrm(a: spy.Array) -> spy.Array:
    return symmetrize_m(symmetrize_lr(a))


def sym(a: spy.Array) -> spy.Array:
    out = 0*a
    p = list(permutations(list(range(a.rank()))))
    for i in p:
        out += spy.permutedims(a, i)
    return out/len(p)


def sym_r(a: spy.Array) -> spy.Array:
    return (a + tt(a, (0, 1, 3, 2)))/2


def sym_l(a: spy.Array) -> spy.Array:
    return (a + tt(a, (1, 0, 2, 3)))/2


def sym_lr(a: spy.Array) -> spy.Array:
    return sym_r(sym_l(a))


#%% Rotations
    
def rotation_matrix(
    n: spy.Array, 
    phi: float
    ) -> spy.Array:
    n = n/nf(n)
    return spy.cos(phi)*ID_2 - spy.sin(phi)*lm(PT, n) + (1-spy.cos(phi))*tp(n, n)


#%% Harmonic tensors

def harmonic(n: int, s: str = "h") -> spy.Array:
    if n==0:
        return spy.symbols(s)
    elif n==1:
        return t(1, s)
    else:
        a = symmetrize(t(n, s))
        return a.subs(linsolve_t(lm(a, ID_2)))


def harmonic_basis(n: int) -> spy.Array:
    h = harmonic(n)
    vs = list(h.free_symbols)
    return spy.Array([spy.diff(h, v) for v in vs])


def harmonic_onb(n: int) -> list:
    return orthogonalize_basis(harmonic_basis(n))


# %% Projections

def project_on_basis(
    a: spy.Array,
    basis: spy.Array
    ) -> spy.Array:
    cs = spy.Array(spy.symbols(f"c:{basis.shape[0]}")) 
    lc = td(cs, basis, 1)
    return cs.subs(linsolve_t(a - lc, cs))


# %% Algebraic decompositions

def iso(a: spy.Array) -> spy.Array:
    if a.rank() == 2:
        return (a[0, 0] + a[1, 1] + a[2, 2])/3*ID_2
    elif a.rank() == 4:
        ls = [sp(a, P)/sp(P, P) for P in P_ISO]
        return ls[0]*P_ISO_1 + ls[1]*P_ISO_2 + ls[2]*P_ISO_3
    else:
        raise Exception("Not implemented")


def aniso(a: spy.Array) -> spy.Array:
    return a - iso(a)


def skw(a: spy.Array) -> spy.Array:
    return a - sym(a)


# %% Differentiation

def grad(a: spy.Array, x: spy.Array) -> spy.Array:
    temp = spy.derive_by_array(a, x)
    ra = a.rank()
    rx = x.rank()
    axes_x = list(range(rx))
    axes_a = list(range(rx, rx + ra))
    return tt(temp, axes_a + axes_x)


# %% Module constants

# Scalars
SR2 = spy.sqrt(2)

# Identity on vectors
ID_2 = spy.Array(spy.eye(3))

# Permutation tensor
PT = spy.MutableDenseNDimArray(
    np.zeros([3, 3, 3], dtype=int)
    )
PT[0,1,2] = 1
PT[1,2,0] = 1
PT[2,0,1] = 1
PT[0,2,1] = -1
PT[2,1,0] = -1
PT[1,0,2] = -1

# Fourth-order
ITI = tp(ID_2, ID_2)
ID_4 = tt(ITI, (0, 2, 1, 3))
ID_S = sym_r(ID_4)
ID_A = ID_4 - ID_S
P_ISO_1 = ITI/3
P_ISO_2 = ID_S - P_ISO_1
P_ISO_3 = ID_A
P_ISO = [P_ISO_1, P_ISO_2, P_ISO_3]
