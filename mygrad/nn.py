#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple implementation of a multi-layer perceptron.
"""

from numbers import Integral
from typing import Tuple

import numpy as np

try:
    from jax import numpy as jnp
    from jax._src.interpreters.ad import JVPTracer

    JAX_AVAILABLE = True

except ImportError:
    JAX_AVAILABLE = False


def relu(x):
    """Relu function"""
    if JAX_AVAILABLE and isinstance(x, JVPTracer):
        return jnp.maximum(x, 0)
    return x.maximum(0)


def softmax(x, axis=-1):
    """Compute softmax values in x by the given axis"""
    if JAX_AVAILABLE and isinstance(x, JVPTracer):
        return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=axis, keepdims=True)

    return x.exp() / x.exp().sum(axis=axis, keepdims=True)


def dense_layer(x, w, b):
    """Dense layer function"""
    return x @ w + b


class MultiLayerPerceptron:
    """
    Axis convention:
    0: batch
    1: feature
    """

    @classmethod
    def initialize(
        self,
        n_input: Integral,
        n_hidden: Tuple[Integral, ...],
        n_output: Integral,
        rng=np.random.randn,
    ):
        return [
            (
                rng(i, o),  # weights
                rng(1, o),  # biases
            ) for i, o in zip(
                [n_input, *n_hidden],
                [*n_hidden, n_output]
            )
        ]

    @classmethod
    def forward(self, x, parameters):
        for w, b in parameters:
            x = relu(dense_layer(x, w, b))
        return softmax(x, axis=1)
