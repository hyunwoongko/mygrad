# mygrad: A Library for Forward-Mode Automatic Differentiation

- Date: 2024/06/23
- Author: Kevin Ko

## 1. Prerequisites

To get started, make sure you have Python version 3.8 or higher. You can install the necessary dependencies with the following command:

```bash
pip install -e .
```

## 2. Supported Features
### 2.1. Forward Pass
You can define a custom forward pass function using `@mg.custom_fn`. Note that this only supports the forward pass, not the backward pass.

```python
import mygrad as mg
import numpy as np

@mg.custom_fn
def fn(x):
    y = (-2.0 * x).exp()
    return (1.0 - y) / (1.0 + y)

x = np.random.rand(2, 2)
output = fn(x)
```

### 2.2. Jacobian-Vector Product
You can compute the Jacobian-vector product and the function output using `mg.jvp`. The usage of this function is inspired by `jax.jvp`.

```python
import mygrad as mg
import numpy as np

def fn(x, y):
    return ((x - y) ** 2).sum().sqrt()

x = np.random.randn(2, 2)
y = np.random.randn(2, 2)
x_grad = np.random.randn(2, 2)
y_grad = np.random.randn(2, 2)

output, grads = mg.jvp(fn, (x, y), (x_grad, y_grad))
```

### 2.3. Graph Tracing
You can trace the computational graph of a function using `mg.trace`.

```python

import mygrad as mg
import numpy as np
import json

def fn(x):
    return (x.sin() ** 2).sum()

x = np.random.randn(2, 2)
graph = mg.trace(fn, x)
json.dump(graph, open("graph_fn.json", "w"), indent=2)
```
The output graph will look something like this:

<details>
<summary>Click to expand</summary>
    
    {
      "kind": "reduce",
      "op_name": "sum",
      "operands": [
        {
          "kind": "binary",
          "op_name": "power",
          "operands": [
            {
              "kind": "unary",
              "op_name": "sin",
              "operands": [
                {
                  "kind": "tensor",
                  "index": 0,
                  "shape": [
                    2,
                    2
                  ],
                  "dtype": "float64",
                  "x": null
                }
              ],
              "args": [],
              "kwargs": {}
            },
            {
              "kind": "python-scalar",
              "index": "P0",
              "shape": [],
              "dtype": "int64",
              "x": null
            }
          ],
          "args": [],
          "kwargs": {}
        }
      ],
      "args": [],
      "kwargs": {}
    }
</details>

Note that some optional arguments of `mg.trace` were added:

- `allow_conditional_flow`: Determines if conditional operations are allowed. If set to `"auto"`, it will check there is any conditional flow in the function and decide to use it. Currently, it only supports static conditional flow.
- `store_forward_pass_output`: Determines if the forward pass output is stored in the graph. Use `dill` instead of `json` to save the graph if this option is enabled because `json` module can't serialize the numpy array.

### 2.4. Graph Execution
You can execute the computational graph using `mg.execute`.

```python
import mygrad as mg
import numpy as np

def fn(x):
    return (x.sin() ** 2).sum()

x = np.random.randn(2, 2)
graph = mg.trace(fn, x)
output = mg.execute(graph, (x,))
```

To compute the Jacobian-vector product together, pass the gradient tensors to `mg.execute`.

```python
import mygrad as mg
import numpy as np

def fn(x):
    return (x.sin() ** 2).sum()

x = np.random.randn(2, 2)
x_grad = np.random.randn(2, 2)

graph = mg.trace(fn, x)
output, grads = mg.execute(graph, (x,), (x_grad,))
```

## 3. Testing
There are two test files: `test_ops.py` and `test_readme.py`.

- `test_ops.py`: Unit tests for operations in `mygrad/ops.py`.
- `test_readme.py`: Tests the code snippets in the `README.md` file.

Run all tests with:

```bash
pytest -vv
```