from __future__ import annotations

import typing

import sympy

if typing.TYPE_CHECKING:
    import sympy


class SympyLib:
    # functions modified specifically for sympy
    def nan_to_num(self, val: sympy.Expr, **kwargs: typing.Any) -> sympy.Expr:
        return val

    def maximum(self, val1: sympy.Expr, val2: sympy.Expr) -> sympy.Expr:
        if val2 == 0:
            return val1
        else:
            return sympy.maximum(val1, val2)  # type: ignore[no-untyped-call]

    def arcsin(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.asin(val)

    def arccos(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.acos(val)

    def arctan(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.atan(val)

    def arctan2(self, val1: sympy.Expr, val2: sympy.Expr) -> sympy.Expr:
        return sympy.atan2(val1, val2)

    def arcsinh(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.asinh(val)

    def arccosh(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.acosh(val)

    def arctanh(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.atanh(val)

    def absolute(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.Abs(val)

    def isclose(
        self,
        val1: sympy.Expr,
        val2: sympy.Expr,
        *args: float,
    ) -> sympy.Equality:
        return sympy.Eq(val1, val2)  # type: ignore[no-untyped-call]

    def copysign(self, val1: sympy.Expr, val2: sympy.Expr) -> sympy.Expr:
        return val1

    def sign(self, val: sympy.Expr) -> sympy.Expr:
        return val

    @property
    def inf(self) -> float | sympy.Expr:
        return sympy.oo

    # same named functions
    def sqrt(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.sqrt(val)  # type: ignore[no-untyped-call]

    def exp(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.exp(val)

    def log(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.log(val)

    def sin(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.sin(val)

    def cos(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.cos(val)

    def tan(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.tan(val)

    def sinh(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.sinh(val)

    def cosh(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.cosh(val)

    def tanh(self, val: sympy.Expr) -> sympy.Expr:
        return sympy.tanh(val)
