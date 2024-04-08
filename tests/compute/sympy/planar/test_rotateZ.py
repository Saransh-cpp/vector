# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

x, y, rho, phi = sympy.symbols("x y rho phi")


def test_xy():
    vec = vector.VectorSympy2D(azimuthal=vector.backends.sympy.AzimuthalSympyXY(x, y))
    assert vec.rotateZ(1).x == x * sympy.cos(1) - y * sympy.sin(1)
    assert vec.rotateZ(1).y == x * sympy.sin(1) + y * sympy.cos(1)


def test_rhophi():
    vec = vector.VectorSympy2D(
        azimuthal=vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi)
    )
    assert vec.rotateZ(1).rho == rho
    assert vec.rotateZ(1).phi == sympy.Mod(phi + 1 + sympy.pi, 2 * sympy.pi) - sympy.pi
