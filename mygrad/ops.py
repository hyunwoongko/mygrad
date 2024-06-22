#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains operations for forward-mode automatic differentiation.
Almost all operations were tested by `tests/test_ops.py` except for some conditional operations like `isinf`, `isnan`, etc.
Actually they don't have gradients, so they are not used in the tests.
"""

import sys

import numpy as np

_FLOAT_MAX = sys.float_info.max
_FLOAT_MIN = sys.float_info.min

_JVP_MAP = {
    # binary operations
    "add": {
        "kind": "binary",
        "magic": ["__add__", "__radd__"],
        "fwd": lambda a, b: a.x + b.x,
        "bwd": lambda a, b: a.g + b.g,
    },
    "multiply": {
        "kind": "binary",
        "magic": ["__mul__", "__rmul__"],
        "fwd": lambda a, b: a.x * b.x,
        "bwd": lambda a, b: a.x * b.g + a.g * b.x,
    },
    "divide": {
        "kind": "binary",
        "magic": ["__truediv__"],
        "fwd": lambda a, b: a.x / b.x,
        "bwd": lambda a, b: (a.g * b.x - a.x * b.g) / (b.x ** 2),
    },
    "rdivide": {
        "kind": "binary",
        "magic": ["__rtruediv__"],
        "fwd": lambda a, b: b.x / a.x,
        "bwd": lambda a, b: (b.g * a.x - b.x * a.g) / (a.x ** 2),
    },
    "power": {
        "kind": "binary",
        "magic": ["__pow__"],
        "fwd": lambda a, b: a.x ** b.x,
        "bwd": lambda a, b: a.x ** b.x * (b.x * a.g / a.x + b.g * np.log(np.abs(a.x))),
    },
    "rpower": {
        "kind": "binary",
        "magic": ["__rpow__"],
        "fwd": lambda a, b: b.x ** a.x,
        "bwd": lambda a, b: b.x ** a.x * (np.log(b.x) * a.g + a.x * b.g / b.x),
    },
    "subtract": {
        "kind": "binary",
        "magic": ["__sub__"],
        "fwd": lambda a, b: a.x - b.x,
        "bwd": lambda a, b: a.g - b.g,
    },
    "rsubtract": {
        "kind": "binary",
        "magic": ["__rsub__"],
        "fwd": lambda a, b: b.x - a.x,
        "bwd": lambda a, b: b.g - a.g,
    },
    "matmul": {
        "kind": "binary",
        "magic": ["__matmul__"],
        "fwd": lambda a, b: a.x @ b.x,
        "bwd": lambda a, b: a.x @ b.g + a.g @ b.x,
    },
    "rmatmul": {
        "kind": "binary",
        "magic": ["__rmatmul__"],
        "fwd": lambda a, b: b.x @ a.x,
        "bwd": lambda a, b: b.x @ a.g + b.g @ a.x,
    },
    "maximum": {
        "kind": "binary",
        "fwd": lambda a, b: np.maximum(a.x, b.x),
        "bwd": lambda a, b: np.where(a.x > b.x, a.g, b.g),
    },
    "minimum": {
        "kind": "binary",
        "fwd": lambda a, b: np.minimum(a.x, b.x),
        "bwd": lambda a, b: np.where(a.x < b.x, a.g, b.g),
    },
    "greater": {
        "kind": "binary",
        "magic": ["__gt__"],
        "fwd": lambda a, b: a.x > b.x,
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "less": {
        "kind": "binary",
        "magic": ["__lt__"],
        "fwd": lambda a, b: a.x < b.x,
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "equal": {
        "kind": "binary",
        "magic": ["__eq__"],
        "fwd": lambda a, b: a.x == b.x,
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "not_equal": {
        "kind": "binary",
        "magic": ["__ne__"],
        "fwd": lambda a, b: a.x != b.x,
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "greater_equal": {
        "kind": "binary",
        "magic": ["__ge__"],
        "fwd": lambda a, b: a.x >= b.x,
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "less_equal": {
        "kind": "binary",
        "magic": ["__le__"],
        "fwd": lambda a, b: a.x <= b.x,
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "logical_and": {
        "kind": "binary",
        "fwd": lambda a, b: np.logical_and(a.x, b.x),
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "logical_or": {
        "kind": "binary",
        "fwd": lambda a, b: np.logical_or(a.x, b.x),
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    "logical_xor": {
        "kind": "binary",
        "fwd": lambda a, b: np.logical_xor(a.x, b.x),
        "bwd": lambda a, b: np.zeros_like(a.x),
    },
    # conditional operations
    "bool": {
        "kind": "cond",
        "magic": ["__bool__"],
        "fwd": lambda a: bool(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    # unary operations
    "neg": {
        "kind": "unary",
        "magic": ["__neg__"],
        "fwd": lambda a: -a.x,
        "bwd": lambda a: -a.g,
    },
    "negative": {
        "kind": "unary",
        "magic": ["__neg__"],
        "fwd": lambda a: -a.x,
        "bwd": lambda a: -a.g,
    },
    "all": {
        "kind": "unary",
        "fwd": lambda a, axis=None: np.all(a.x, axis=axis),
        "bwd": lambda a, axis=None: np.zeros_like(a.x),
    },
    "any": {
        "kind": "unary",
        "fwd": lambda a, axis=None: np.any(a.x, axis=axis),
        "bwd": lambda a, axis=None: np.zeros_like(a.x),
    },
    "isnan": {
        "kind": "unary",
        "fwd": lambda a: np.isnan(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "isinf": {
        "kind": "unary",
        "fwd": lambda a: np.isinf(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "isfinite": {
        "kind": "unary",
        "fwd": lambda a: np.isfinite(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "logical_not": {
        "kind": "unary",
        "fwd": lambda a: np.logical_not(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "abs": {
        "kind": "unary",
        "fwd": lambda a: np.abs(a.x),
        "bwd": lambda a: a.g * np.sign(a.x),
    },
    "absolute": {
        "kind": "unary",
        "fwd": lambda a: np.abs(a.x),
        "bwd": lambda a: a.g * np.sign(a.x),
    },
    "log": {
        "kind": "unary",
        "fwd": lambda a: np.log(a.x),
        "bwd": lambda a: a.g / a.x,
    },
    "log2": {
        "kind": "unary",
        "fwd": lambda a: np.log2(a.x),
        "bwd": lambda a: a.g / (a.x * np.log(2)),
    },
    "log10": {
        "kind": "unary",
        "fwd": lambda a: np.log10(a.x),
        "bwd": lambda a: a.g / (a.x * np.log(10)),
    },
    "log1p": {
        "kind": "unary",
        "fwd": lambda a: np.log1p(a.x),
        "bwd": lambda a: a.g / (1 + a.x),
    },
    "exp": {
        "kind": "unary",
        "fwd": lambda a: np.exp(a.x),
        "bwd": lambda a: a.g * np.exp(a.x),
    },
    "expm1": {
        "kind": "unary",
        "fwd": lambda a: np.expm1(a.x),
        "bwd": lambda a: a.g * np.exp(a.x),
    },
    "sqrt": {
        "kind": "unary",
        "fwd": lambda a: np.sqrt(a.x),
        "bwd": lambda a: a.g / (2 * np.sqrt(a.x)),
    },
    "cbrt": {
        "kind": "unary",
        "fwd": lambda a: np.cbrt(a.x),
        "bwd": lambda a: a.g / (3 * np.cbrt(a.x) ** 2),
    },
    "square": {
        "kind": "unary",
        "fwd": lambda a: np.square(a.x),
        "bwd": lambda a: 2 * a.x * a.g,
    },
    "reciprocal": {
        "kind": "unary",
        "fwd": lambda a: 1 / a.x,
        "bwd": lambda a: -a.g / (a.x ** 2),
    },
    "sign": {
        "kind": "unary",
        "fwd": lambda a: np.sign(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "floor": {
        "kind": "unary",
        "fwd": lambda a: np.floor(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "ceil": {
        "kind": "unary",
        "fwd": lambda a: np.ceil(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "trunc": {
        "kind": "unary",
        "fwd": lambda a: np.trunc(a.x),
        "bwd": lambda a: np.zeros_like(a.x),
    },
    "sin": {
        "kind": "unary",
        "fwd": lambda a: np.sin(a.x),
        "bwd": lambda a: a.g * np.cos(a.x),
    },
    "cos": {
        "kind": "unary",
        "fwd": lambda a: np.cos(a.x),
        "bwd": lambda a: -a.g * np.sin(a.x),
    },
    "tan": {
        "kind": "unary",
        "fwd": lambda a: np.tan(a.x),
        "bwd": lambda a: a.g / np.cos(a.x) ** 2,
    },
    "arcsin": {
        "kind": "unary",
        "fwd": lambda a: np.arcsin(a.x),
        "bwd": lambda a: a.g / np.sqrt(1 - a.x ** 2),
    },
    "arccos": {
        "kind": "unary",
        "fwd": lambda a: np.arccos(a.x),
        "bwd": lambda a: -a.g / np.sqrt(1 - a.x ** 2),
    },
    "arctan": {
        "kind": "unary",
        "fwd": lambda a: np.arctan(a.x),
        "bwd": lambda a: a.g / (1 + a.x ** 2),
    },
    "sinh": {
        "kind": "unary",
        "fwd": lambda a: np.sinh(a.x),
        "bwd": lambda a: a.g * np.cosh(a.x),
    },
    "cosh": {
        "kind": "unary",
        "fwd": lambda a: np.cosh(a.x),
        "bwd": lambda a: a.g * np.sinh(a.x),
    },
    "tanh": {
        "kind": "unary",
        "fwd": lambda a: np.tanh(a.x),
        "bwd": lambda a: a.g / np.cosh(a.x) ** 2,
    },
    "arcsinh": {
        "kind": "unary",
        "fwd": lambda a: np.arcsinh(a.x),
        "bwd": lambda a: a.g / np.sqrt(a.x ** 2 + 1),
    },
    "arccosh": {
        "kind": "unary",
        "fwd": lambda a: np.arccosh(a.x),
        "bwd": lambda a: a.g / np.sqrt(a.x ** 2 - 1),
    },
    "arctanh": {
        "kind": "unary",
        "fwd": lambda a: np.arctanh(a.x),
        "bwd": lambda a: a.g / (1 - a.x ** 2),
    },
    # elementwise operations
    "clip": {
        "kind": "elementwise",
        "fwd": lambda a, a_min=-_FLOAT_MAX, a_max=_FLOAT_MAX: np.clip(a.x, a_min, a_max),
        "bwd": lambda a, a_min=-_FLOAT_MAX, a_max=_FLOAT_MAX: np.where((a.x > a_min) & (a.x < a_max), a.g, 0),
    },
    "transpose": {
        "kind": "elementwise",
        "fwd": lambda a, axes=None: np.transpose(a.x, axes=axes),
        "bwd": lambda a, axes=None: np.transpose(a.g, axes=axes),
    },
    "T": {
        "kind": "elementwise",
        "property": True,
        "fwd": lambda a: a.x.T,
        "bwd": lambda a: a.g.T,
    },
    # reduce operations
    "sum": {
        "kind": "reduce",
        "fwd": lambda a, axis=None, keepdims=False: np.sum(a.x, axis=axis, keepdims=keepdims),
        "bwd": lambda a, axis=None, keepdims=False: np.sum(a.g, axis=axis, keepdims=keepdims),
    },
    "mean": {
        "kind": "reduce",
        "fwd": lambda a, axis=None, keepdims=False: np.mean(a.x, axis=axis, keepdims=keepdims),
        "bwd": lambda a, axis=None, keepdims=False: np.mean(a.g, axis=axis, keepdims=keepdims),
    },
    "max": {
        "kind": "reduce",
        "fwd": lambda a, axis=None: np.max(a.x, axis=axis),
        "bwd": lambda a, axis=None: np.take_along_axis(
            (a.x == np.expand_dims(np.max(a.x, axis=axis), axis)) * a.g,
            np.argmax(a.x, axis=axis, keepdims=True),
            axis=axis,
        ).squeeze(axis=axis) if axis is not None else a.g.flat[np.argmax(a.x)],
    },
    "min": {
        "kind": "reduce",
        "fwd": lambda a, axis=None: np.min(a.x, axis=axis),
        "bwd": lambda a, axis=None: np.take_along_axis(
            (a.x == np.expand_dims(np.min(a.x, axis=axis), axis)) * a.g,
            np.argmin(a.x, axis=axis, keepdims=True),
            axis=axis,
        ).squeeze(axis=axis) if axis is not None else a.g.flat[np.argmin(a.x)],
    },
}
