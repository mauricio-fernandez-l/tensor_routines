# %% Import

import numpy as np
import tensor_routines.numpy_routines as tn

# %%

tol = 1e-13

# %%

def test_td():
    a = np.random.rand(2, 3, 4, 5)
    b = np.random.rand(4, 5, 6, 7)
    c = tn.td(a, b, 2)
    assert c.shape == (2, 3, 6, 7)


def test_sp():
    for _ in range(100):
        a = np.random.rand(2, 3, 4)
        b = np.random.rand(2, 3, 4)
        c = tn.sp(a, b)
        if abs(c - np.sum(a.flatten()*b.flatten())) >= tol:
            break
    assert abs(c - np.sum(a.flatten()*b.flatten())) < tol


def test_nf():
    for _ in range(100):
        a = np.random.rand(2, 3, 4)
        if abs(tn.nf(a) - np.sqrt(np.sum(a**2))) >= tol:
            break
    assert abs(tn.nf(a) - np.sqrt(np.sum(a**2))) < tol


def test_tp():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(7, 8, 9)
    c = tn.tp(a, b) # c_ijkmno = a_ijk b_mno
    assert c.shape == (2, 3, 4, 7, 8, 9)
    assert tn.nf(c[0, 1, 2] - a[0, 1, 2]*b) < tol
    assert tn.tp(a, a, b).shape == (2, 3, 4, 2, 3, 4, 7, 8, 9)


def test_lm():
    a = np.random.rand(2, 3, 4, 5, 6)
    b = np.random.rand(4, 5, 6)
    c = tn.lm(a, b)
    assert c.shape == (2, 3)
    assert tn.nf(c[0, 1] - tn.sp(a[0, 1], b))/tn.nf(c[0, 1]) < tol


def test_rp():
    d_in = 5
    d_out = 7
    q = np.random.rand(d_out, d_in)
    shape_in = [d_in]*4
    shape_out = [d_out]*4
    a = np.random.rand(*shape_in)
    b = tn.rp(q, a)
    b_ = np.zeros(shape_out)
    for i0 in range(d_out):
        for i1 in range(d_out):
            for i2 in range(d_out):
                for i3 in range(d_out):
                    b_[i0, i1, i2, i3] = tn.sp(tn.tp(q[i0], q[i1], q[i2], q[i3]), a)
    assert tn.nf(b - b_)/tn.nf(b) < tol


def test_tt():
    a = np.random.rand(2, 3, 4, 5)
    assert tn.tt(a, (2, 0, 3, 1)).shape == (4, 2, 5, 3)


def test_sym_lr():
    a = np.random.rand(3, 3, 3, 3)
    b = tn.sym_lr(a)
    assert (b == tn.tt_r(b)).all()
    assert (b == tn.tt_l(b)).all()


def test_iso():
    # iso_t
    assert (tn.iso_t(10) == 10*np.eye(3)).all()
    assert (tn.iso_t([3, 7]) == 3*tn.P_ISO_1 + 7*tn.P_ISO_2).all()
    # iso_proj and iso_ev
    ls = np.array([11, 7, 13])
    a = tn.iso_t(ls)
    assert tn.nf(a - tn.iso_proj(a))/tn.nf(a) < tol
    # iso_inv
    a = tn.iso_t(7)
    ai = tn.iso_inv(a)
    assert tn.nf(a@ai - np.eye(3)) < tol
    ls = np.array([11, 7, 13])
    a = tn.iso_t(ls)
    ai = tn.iso_inv(a)
    assert tn.nf(tn.td(a, ai, 2) - tn.ID_4)/tn.nf(tn.ID_S) < tol
    ls = np.array([11, 7, 0])
    a = tn.iso_t(ls)
    ai = tn.iso_inv(a)
    assert tn.nf(tn.td(a, ai, 2) - tn.ID_S)/tn.nf(tn.ID_S) < tol
    

def test_nvn():
    # Second order
    a = tn.sym_2(np.random.rand(3, 3))
    e = tn.nf(a - tn.nvn_inv(tn.nvn(a)))/tn.nf(a)
    assert e < tol
    # Fourth order
    a = tn.sym_lr(np.random.rand(3, 3, 3, 3))
    e = tn.nf(a - tn.nvn_inv(tn.nvn(a)))/tn.nf(a)
    assert e < tol
    # inv_nvn
    a = tn.iso_t([1, 2, 0])
    ai = tn.inv_nvn(a)
    e = tn.nf(tn.td(a, ai, 2) - tn.ID_S)/tn.nf(tn.ID_S)
    assert e < tol


def test_tric():
    a = tn.t4_tric(*np.random.rand(21))
    assert (a == tn.tt_l(a)).all()
    assert (a == tn.tt_r(a)).all()


def test_hex():
    a = tn.t4_hex(*np.random.random(5))
    for _ in range(100):
        Q = tn.rotation_matrix(
            np.array([0, 0, 1]), 
            np.random.rand()*np.pi
            )
        b = tn.rp(Q, a)
        if tn.nf(a - b)/tn.nf(a) >= tol:
            break 
    assert tn.nf(a - b)/tn.nf(a) < tol


