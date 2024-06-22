#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The main script for the forward-mode automatic differentiation home assignment.
"""

import json

import jax
import numpy as np

import mygrad as mg
from mygrad.nn import MultiLayerPerceptron

# Network related constants
N_INPUT = 2
N_OUTPUT = 3
N_HIDDEN = (20, 10)

# Input and gradient related constants
INPUT_SIZE = 10
GRAD_SIZE = 4


def compute_gradients(impl, func, input_data, input_gradients):
    output_gradients = np.zeros((INPUT_SIZE, N_OUTPUT, GRAD_SIZE))
    for idx in range(GRAD_SIZE):
        outputs, grads = impl(func, (input_data,), (input_gradients[..., idx],))
        output_gradients[..., idx] = grads
    return output_gradients


def main():
    np.random.seed(200527)
    input_data = np.random.randn(INPUT_SIZE, N_INPUT)
    model_parameters = MultiLayerPerceptron.initialize(
        n_input=N_INPUT,
        n_hidden=N_HIDDEN,
        n_output=N_OUTPUT,
        rng=lambda *shape: np.random.randn(*shape) * 0.1
    )
    model_forward = lambda x: MultiLayerPerceptron.forward(
        x, model_parameters
    )

    eager_result = mg.custom_fn(model_forward)(input_data)
    print('Eager-mode forward calculation:\n\n', eager_result, '\n')

    graph = mg.trace(model_forward, input_data)
    # print("Computational graph:\n\n", json.dumps(graph), '\n')

    interpreted_result = mg.execute(graph, (input_data,))
    print("Graph execution of forward calculation:\n\n", interpreted_result, '\n')

    # derivative of the input data w.r.t a 4-dimensional parameter vector
    # TODO: transform `graph` into another graph which when executed will
    # compute the derivative of the network output w.r.t. to the derivative
    # of the input date using forward differentiation
    input_gradients = np.random.randn(*input_data.shape, GRAD_SIZE)

    # 1. Compute output gradients from the graph.
    output_gradients_by_graph = compute_gradients(mg.execute, graph, input_data, input_gradients)

    # 2. Compute output gradients using eager mode.
    output_gradients_by_eager = compute_gradients(mg.jvp, model_forward, input_data, input_gradients)

    # 3. Compute output gradients using jax.jvp.
    output_gradients_by_jax = compute_gradients(jax.jvp, model_forward, input_data, input_gradients)

    # 4. Check difference between the results
    print("Comparison of the gradients calculation:")
    print("- graph vs eager   =>", np.allclose(output_gradients_by_graph, output_gradients_by_eager))
    print("- graph vs jax.jvp =>", np.allclose(output_gradients_by_graph, output_gradients_by_jax))
    print("- eager vs jax.jvp =>", np.allclose(output_gradients_by_eager, output_gradients_by_jax))


if __name__ == '__main__':
    main()

# Output:
#
# Eager-mode forward calculation:
#
#  [[0.30931468 0.30931468 0.38137064]
#  [0.30655935 0.30655935 0.38688129]
#  [0.30768159 0.30768159 0.38463682]
#  [0.30791452 0.30791452 0.38417095]
#  [0.30729894 0.30729894 0.38540211]
#  [0.30876927 0.30876927 0.38246146]
#  [0.31022946 0.31022946 0.37954107]
#  [0.3077649  0.3077649  0.38447019]
#  [0.3072738  0.3072738  0.38545241]
#  [0.309631   0.309631   0.380738  ]]
#
# Graph execution of forward calculation:
#
#  [[0.30931468 0.30931468 0.38137064]
#  [0.30655935 0.30655935 0.38688129]
#  [0.30768159 0.30768159 0.38463682]
#  [0.30791452 0.30791452 0.38417095]
#  [0.30729894 0.30729894 0.38540211]
#  [0.30876927 0.30876927 0.38246146]
#  [0.31022946 0.31022946 0.37954107]
#  [0.3077649  0.3077649  0.38447019]
#  [0.3072738  0.3072738  0.38545241]
#  [0.309631   0.309631   0.380738  ]]
#
# Comparison of the gradients calculation:
# - graph vs eager   => True
# - graph vs jax.jvp => True
# - eager vs jax.jvp => True
