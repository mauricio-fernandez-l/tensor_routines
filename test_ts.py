# %% Import

import sympy as spy
import numpy as np
import tensor_routines.sympy_routines as ts

# %%

def test_nf():
    a = ts.t(3)
    assert spy.sqrt(ts.sp(a, a)) == ts.nf(a)


def test_rp():
    # First order
    a = ts.t(1)
    q = ts.t(2, "q")
    assert spy.simplify()
    # Second order
    a = ts.t(2)
    q = ts.t(2, "q")
    temp_1 = ts.rp(q, a).tomatrix()
    a = a.tomatrix()
    q = q.tomatrix()
    temp_2 = q*a*spy.transpose(q)
    assert spy.simplify(temp_1 - temp_2) == spy.zeros(3, 3)
    # Higher order
    a = ts.t(2, "a", [2, 2, 2])
    q = ts.t(2, "q", [4, 2])
    temp_1 = ts.rp(q, a)
    d_out = q.shape[0]
    zeros = spy.MutableDenseNDimArray(
        np.zeros([d_out]*a.rank(), dtype=int)
        )
    temp_2 = spy.MutableDenseNDimArray(
        np.zeros([d_out]*a.rank(), dtype=int)
        )
    for i0 in range(d_out):
        for i1 in range(d_out):
            for i2 in range(d_out):
                    temp_2[i0, i1, i2] = ts.sp(ts.tp(q[i0], q[i1], q[i2]), a)
    assert spy.simplify(temp_1 - temp_2) == zeros


def test_lm():
    # Scalar output
    a = ts.t(3, "a", [2, 3, 4])
    b = ts.t(3, "b", [2, 3, 4])
    assert ts.lm(a, b).is_scalar
    # Tensor output
    a = ts.t(5, "a")
    b = ts.t(2, "b")
    assert ts.lm(a, b).shape == (3, 3, 3)


def test_nvn():
    a = ts.symmetrize_lr(ts.t(4))
    b = ts.symmetrize(ts.t(2, "b"))
    c = ts.lm(a, b)
    check = spy.simplify(ts.nvn(c) - ts.nvn(a)*ts.nvn(b)) == spy.zeros(6, 1)
    assert check


def test_grad():
    a = ts.t(2, "a")
    b = ts.t(2, "b")
    temp = ts.grad(ts.tp(a, b), b)
    assert temp == ts.tp(a, ts.ID_4)