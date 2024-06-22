def test_forward_pass():
    import mygrad as mg
    import numpy as np

    @mg.custom_fn
    def fn(x):
        y = (-2.0 * x).exp()
        return (1.0 - y) / (1.0 + y)

    # Inputs
    x = np.random.rand(2, 2)

    # Do forward pass
    output = fn(x)


def test_jacobian_vector_product():
    import mygrad as mg
    import numpy as np

    def fn(x, y):
        return ((x - y) ** 2).sum().sqrt()

    x = np.random.randn(2, 2)
    y = np.random.randn(2, 2)
    x_grad = np.random.randn(2, 2)
    y_grad = np.random.randn(2, 2)

    # Compute the function output and Jacobian-vector product
    output, grads = mg.jvp(fn, (x, y), (x_grad, y_grad))


def test_graph_tracing():
    import mygrad as mg
    import numpy as np
    import json

    def fn(x):
        return (x.sin() ** 2).sum()

    x = np.random.randn(2, 2)
    graph = mg.trace(fn, x)
    # json.dump(graph, open("graph_fn.json", "w"), indent=2)


def test_graph_execution():
    import mygrad as mg
    import numpy as np

    def fn(x):
        return (x.sin() ** 2).sum()

    x = np.random.randn(2, 2)
    graph = mg.trace(fn, x)
    output = mg.execute(graph, (x,))

    x = np.random.randn(2, 2)
    x_grad = np.random.randn(2, 2)

    graph = mg.trace(fn, x)
    output, grads = mg.execute(graph, (x,), (x_grad,))

