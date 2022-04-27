# tensor_routines

Information:

* Author: Dr.-Ing. Mauricio Fernández
* Contact: mauricio.fernandez.lb@gmail.com
* Last edited: 2022-04-27
* [GitHub](https://github.com/mauricio-fernandez-l/tensor_routines)

Personal collection of `numpy` and `sympy` routines for computations with tensors in continuum mechanics and other areas.

## Installation

Install with pip

```shell
pip install .
```

## Usage

Create a numeric stiffness tensor of an isotropic or cubic material based on its 2 and 3 respective eigenvalues

```python
import tensor_routines.numpy_routines as tn
stiffness_iso = tn.stiffness_iso_l(1, 2)
stiffness_cub = tn.stiffness_cub_l(1, 2, 3)
```

Create a 4th-order symbolic tensor, symmetrize it left and right, and print its normalized Voigt notation matrix

```python
import tensor_routines.sympy_routines as ts
a = ts.t(4, "a")
a = ts.symmetrize_lr(a)
a = ts.nvn(a)
print(a)
```
