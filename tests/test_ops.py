import jax
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters.ad import JVPTracer

import mygrad as mg

# Seed for reproducibility
np.random.seed(200527)

# Random input data
x1 = np.random.randn(3, 3)
x2 = np.random.randn(3, 3)
g1 = np.random.randn(3, 3)
g2 = np.random.randn(3, 3)


# Helper function to compare the forward and backward pass with JAX
def assert_allclose(fn, inputs, grads):
    my_output = mg.jvp(fn, inputs, grads)
    jax_output = jax.jvp(fn, inputs, grads)
    assert np.allclose(my_output[0], jax_output[0], equal_nan=True), \
        f"Forward pass mismatch for {fn.__name__}, {my_output[0]} vs {jax_output[0]}"
    assert np.allclose(my_output[1], jax_output[1], equal_nan=True), \
        f"Backward pass mismatch for {fn.__name__}, {my_output[1]} vs {jax_output[1]}"


# Helper function to check if the object is a JAX tracer
def is_jax(obj):
    return isinstance(obj, JVPTracer)


# Test operations
def test_add():
    def add(a, b):
        return a + b

    assert_allclose(add, (x1, x2), (g1, g2))


def test_multiply():
    def multiply(a, b):
        return a * b

    assert_allclose(multiply, (x1, x2), (g1, g2))


def test_subtract():
    def subtract(a, b):
        return a - b

    assert_allclose(subtract, (x1, x2), (g1, g2))


def test_divide():
    def divide(a, b):
        return a / b

    assert_allclose(divide, (x1, x2), (g1, g2))


def test_rdivide():
    def rdivide(a, b):
        return b / a

    assert_allclose(rdivide, (x1, x2), (g1, g2))


def test_power():
    def power(a):
        return a ** 2

    assert_allclose(power, (x1,), (g1,))


def test_rpower():
    def rpower(a):
        return 2 ** a

    assert_allclose(rpower, (x1,), (g1,))


def test_matmul():
    def matmul(a, b):
        return a @ b

    assert_allclose(matmul, (x1, x2), (g1, g2))


def test_rmatmul():
    def rmatmul(a, b):
        return b @ a

    assert_allclose(rmatmul, (x1, x2), (g1, g2))


def test_maximum():
    def maximum(a, b):
        if is_jax(a):
            return jnp.maximum(a, b)
        return a.maximum(b)

    assert_allclose(maximum, (x1, x2), (g1, g2))


def test_minimum():
    def minimum(a, b):
        if is_jax(a):
            return jnp.minimum(a, b)
        return a.minimum(b)

    assert_allclose(minimum, (x1, x2), (g1, g2))


def test_negative():
    def negative(a):
        return -a

    assert_allclose(negative, (x1,), (g1,))


def test_abs():
    def abs(a):
        if is_jax(a):
            return jnp.abs(a)
        return a.abs()

    assert_allclose(abs, (x1,), (g1,))


def test_log():
    def log(a):
        if is_jax(a):
            return jnp.log(a)
        return a.log()

    assert_allclose(log, (x1,), (g1,))


def test_log2():
    def log2(a):
        if is_jax(a):
            return jnp.log2(a)
        return a.log2()

    assert_allclose(log2, (x1,), (g1,))


def test_log10():
    def log10(a):
        if is_jax(a):
            return jnp.log10(a)
        return a.log10()

    assert_allclose(log10, (x1,), (g1,))


def test_log1p():
    def log1p(a):
        if is_jax(a):
            return jnp.log1p(a)
        return a.log1p()

    assert_allclose(log1p, (x1,), (g1,))


def test_exp():
    def exp(a):
        if is_jax(a):
            return jnp.exp(a)
        return a.exp()

    assert_allclose(exp, (x1,), (g1,))


def test_expm1():
    def expm1(a):
        if is_jax(a):
            return jnp.expm1(a)
        return a.expm1()

    assert_allclose(expm1, (x1,), (g1,))


def test_sqrt():
    def sqrt(a):
        if is_jax(a):
            return jnp.sqrt(a)
        return a.sqrt()

    assert_allclose(sqrt, (x1,), (g1,))


def test_cbrt():
    def cbrt(a):
        if is_jax(a):
            return jnp.cbrt(a)
        return a.cbrt()

    assert_allclose(cbrt, (x1,), (g1,))


def test_square():
    def square(a):
        if is_jax(a):
            return jnp.square(a)
        return a.square()

    assert_allclose(square, (x1,), (g1,))


def test_reciprocal():
    def reciprocal(a):
        if is_jax(a):
            return jnp.reciprocal(a)
        return a.reciprocal()

    assert_allclose(reciprocal, (x1,), (g1,))


def test_sign():
    def sign(a):
        if is_jax(a):
            return jnp.sign(a)
        return a.sign()

    assert_allclose(sign, (x1,), (g1,))


def test_floor():
    def floor(a):
        if is_jax(a):
            return jnp.floor(a)
        return a.floor()

    assert_allclose(floor, (x1,), (g1,))


def test_ceil():
    def ceil(a):
        if is_jax(a):
            return jnp.ceil(a)
        return a.ceil()

    assert_allclose(ceil, (x1,), (g1,))


def test_trunc():
    def trunc(a):
        if is_jax(a):
            return jnp.trunc(a)
        return a.trunc()

    assert_allclose(trunc, (x1,), (g1,))


def test_sin():
    def sin(a):
        if is_jax(a):
            return jnp.sin(a)
        return a.sin()

    assert_allclose(sin, (x1,), (g1,))


def test_cos():
    def cos(a):
        if is_jax(a):
            return jnp.cos(a)
        return a.cos()

    assert_allclose(cos, (x1,), (g1,))


def test_tan():
    def tan(a):
        if is_jax(a):
            return jnp.tan(a)
        return a.tan()

    assert_allclose(tan, (x1,), (g1,))


def test_arcsin():
    def arcsin(a):
        if is_jax(a):
            return jnp.arcsin(a)
        return a.arcsin()

    assert_allclose(arcsin, (x1,), (g1,))


def test_arccos():
    def arccos(a):
        if is_jax(a):
            return jnp.arccos(a)
        return a.arccos()

    assert_allclose(arccos, (x1,), (g1,))


def test_arctan():
    def arctan(a):
        if is_jax(a):
            return jnp.arctan(a)
        return a.arctan()

    assert_allclose(arctan, (x1,), (g1,))


def test_sinh():
    def sinh(a):
        if is_jax(a):
            return jnp.sinh(a)
        return a.sinh()

    assert_allclose(sinh, (x1,), (g1,))


def test_cosh():
    def cosh(a):
        if is_jax(a):
            return jnp.cosh(a)
        return a.cosh()

    assert_allclose(cosh, (x1,), (g1,))


def test_tanh():
    def tanh(a):
        if is_jax(a):
            return jnp.tanh(a)
        return a.tanh()

    assert_allclose(tanh, (x1,), (g1,))


def test_arcsinh():
    def arcsinh(a):
        if is_jax(a):
            return jnp.arcsinh(a)
        return a.arcsinh()

    assert_allclose(arcsinh, (x1,), (g1,))


def test_arccosh():
    def arccosh(a):
        if is_jax(a):
            return jnp.arccosh(a)
        return a.arccosh()

    assert_allclose(arccosh, (x1,), (g1,))


def test_arctanh():
    def arctanh(a):
        if is_jax(a):
            return jnp.arctanh(a)
        return a.arctanh()

    assert_allclose(arctanh, (x1,), (g1,))


def test_clip():
    def clip(a):
        if is_jax(a):
            return jnp.clip(a, 0.1, 0.5)
        return a.clip(0.1, 0.5)

    assert_allclose(clip, (x1,), (g1,))


def test_clip_kwargs():
    def clip(a):
        if is_jax(a):
            return jnp.clip(a, a_min=0.1, a_max=0.5)
        return a.clip(a_min=0.1, a_max=0.5)

    assert_allclose(clip, (x1,), (g1,))


def test_transpose():
    def transpose(a):
        if is_jax(a):
            return jnp.transpose(a)
        return a.transpose()

    assert_allclose(transpose, (x1,), (g1,))


def test_transpose_kwargs():
    def transpose(a):
        if is_jax(a):
            return jnp.transpose(a, axes=(1, 0))
        return a.transpose(axes=(1, 0))

    assert_allclose(transpose, (x1,), (g1,))


def test_sum():
    def sum_(a):
        if is_jax(a):
            return jnp.sum(a)
        return a.sum()

    assert_allclose(sum_, (x1,), (g1,))


def test_sum_kwargs1():
    def sum_(a):
        if is_jax(a):
            return jnp.sum(a, axis=1)
        return a.sum(axis=1)

    assert_allclose(sum_, (x1,), (g1,))


def test_sum_kwargs2():
    def sum_(a):
        if is_jax(a):
            return jnp.sum(a, axis=1, keepdims=True)
        return a.sum(axis=1, keepdims=True)

    assert_allclose(sum_, (x1,), (g1,))


def test_mean():
    def mean(a):
        if is_jax(a):
            return jnp.mean(a)
        return a.mean()

    assert_allclose(mean, (x1,), (g1,))


def test_mean_kwargs():
    def mean(a):
        if is_jax(a):
            return jnp.mean(a, axis=1)
        return a.mean(axis=1)

    assert_allclose(mean, (x1,), (g1,))


def test_mean_kwargs2():
    def mean(a):
        if is_jax(a):
            return jnp.mean(a, axis=1, keepdims=True)
        return a.mean(axis=1, keepdims=True)

    assert_allclose(mean, (x1,), (g1,))


def test_max():
    def max_(a):
        if is_jax(a):
            return jnp.max(a)
        return a.max()

    assert_allclose(max_, (x1,), (g1,))


def test_max_kwargs():
    def max_(a):
        if is_jax(a):
            return jnp.max(a, axis=1)
        return a.max(axis=1)

    assert_allclose(max_, (x1,), (g1,))


def test_min():
    def min_(a):
        if is_jax(a):
            return jnp.min(a)
        return a.min()

    assert_allclose(min_, (x1,), (g1,))


def test_min_kwargs():
    def min_(a):
        if is_jax(a):
            return jnp.min(a, axis=1)
        return a.min(axis=1)

    assert_allclose(min_, (x1,), (g1,))


def test_cond1():
    def cond(a):
        if is_jax(a):
            if jnp.mean(jnp.sin(a)) > 0:
                return a
            else:
                return -a

        if a.sin().mean() > 0:
            return a
        else:
            return -a

    assert_allclose(cond, (x1,), (g1,))


def test_cond2():
    def cond(a):
        if is_jax(a):
            if jnp.any(a > 0.5):
                return a
            else:
                return -a

        if (a > 0.5).any():
            return a
        else:
            return -a

    assert_allclose(cond, (x1,), (g1,))


def test_cond3():
    def cond(a):
        if is_jax(a):
            if jnp.all(a < 1):
                return a ** 2
            else:
                return a / 2

        if (a < 1).all():
            return a ** 2
        else:
            return a / 2

    assert_allclose(cond, (x1,), (g1,))


def test_cond4():
    def cond(a):
        if is_jax(a):
            if jnp.isfinite(jnp.sum(a)):
                return a + 1
            else:
                return a - 1

        if a.sum().isfinite():
            return a + 1
        else:
            return a - 1

    assert_allclose(cond, (x1,), (g1,))


def test_cond5():
    def cond(a):
        if is_jax(a):
            if jnp.max(jnp.abs(a)) > 1:
                return a ** 2 / jnp.max(a)
            else:
                return a ** 2 / jnp.min(a)

        if a.abs().max() > 1:
            return a ** 2 / a.max()
        else:
            return a ** 2 / a.min()

    assert_allclose(cond, (x1,), (g1,))
