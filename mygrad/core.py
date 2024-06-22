#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains the core features about forward-mode automatic differentiation using the JVP.
The dual numbers are used to compute the Jacobian-vector product of a function.
"""

import threading
from typing import Callable, Tuple, Optional, List, Dict, Any

import numpy as np

from mygrad.ops import _JVP_MAP

_TENSOR_INDEX = 0
_PYTHON_SCALAR_INDEX = "P0"
_PYTHON_SCALR_MAP = {}

# This is a single thread program, so the lock is not necessary.
# But I added it to make the code more robust when I extend it
# to a multi thread program in the future.

_TENSOR_INIT_LOCK = threading.Lock()
_PYTHON_SCALAR_INIT_LOCK = threading.Lock()


class _NodeForEagerMode:
    """
    A node that have the primal and the tangent of a function at when eager mode is activated.
    This was implemented based on dual number (https://wikipedia.org/wiki/Dual_number).

    Args:
        x (np.ndarray):
            The input value of the node.
        g (Optional[np.ndarray]):
            The gradient value of the node.
    """

    def __init__(
        self,
        x: np.ndarray,
        g: Optional[np.ndarray] = None,
    ):
        self.x = x
        self.g = g

    @property
    def need_bwd(self):
        """Check if the backward pass is needed."""
        return self.g is not None

    def __repr__(self):
        """Return a string representation of the node."""
        return f"_NodeForEagerMode(x={self.x}, g={self.g})"

    def __str__(self):
        """Return a string representation of the node."""
        return f"_NodeForEagerMode(x={self.x}, g={self.g})"

    def __len__(self):
        """Return the length of the input tensor."""
        return len(self.x)

    def shape(self):
        """Return the shape of the input tensor."""
        return self.x.shape


class _NodeForTracingMode:
    """
    A node that stores the information of a function when tracing mode is activated.

    Args:
        kind (str):
            The kind of the node.
        allow_conditional_flow (Union[str, bool]): Whether to allow conditional operations.
            If "auto", it will be determined based on the function.
            If it's True, forward pass will be done to determine the conditional flow.
        store_forward_pass_output (bool): Whether to store the forward pass output in the graph.
            It can be useful to check which input tensors are used in the forward pass when debugging the graph.
            Use `dill` instead of `json` to serialize the graph when this option is enabled.
        x (Optional[Any]):
            The value of the tensor.
        op_name (Optional[str]):
            The operation name of the node.
        operands (List[_NodeForTracingMode]):
            The operands of the node.
        shape (Optional[Tuple[int, ...]]):
            The shape of the tensor.
        dtype (Optional[str]):
            The data type of the tensor.
        args (Optional[Tuple]):
            The arguments of the operation, not supported for binary operations currently.
        kwargs (Optional[Dict[str, Any]]):
            The keyword arguments of the operation.
    """

    def __init__(
        self,
        kind: str,
        allow_conditional_flow: bool,
        store_forward_pass_output: bool,
        x: Optional[Any] = None,
        op_name: Optional[str] = None,
        operands: Optional[List["_NodeForTracingMode"]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.kind = kind
        self.allow_conditional_flow = allow_conditional_flow
        self.store_forward_pass_output = store_forward_pass_output
        self.x = x
        self.operands = operands
        self.shape = shape
        self.op_name = op_name
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs

        if self.kind == "tensor":
            global _TENSOR_INDEX
            _TENSOR_INIT_LOCK.acquire()
            self.index = _TENSOR_INDEX
            _TENSOR_INDEX += 1
            _TENSOR_INIT_LOCK.release()

        elif self.kind == "python-scalar":
            global _PYTHON_SCALAR_INDEX
            _PYTHON_SCALAR_INIT_LOCK.acquire()
            self.index = _PYTHON_SCALAR_INDEX
            _PYTHON_SCALR_MAP[self.index] = self.x
            number_only = int(_PYTHON_SCALAR_INDEX[1:])
            _PYTHON_SCALAR_INDEX = f"P{number_only + 1}"
            _PYTHON_SCALAR_INIT_LOCK.release()

    @property
    def need_fwd(self):
        """Check if the forward pass is needed."""
        return self.allow_conditional_flow or self.store_forward_pass_output

    def to_dict(self):
        """
        Convert the node to a dictionary.

        Returns:
            Dict[str, Any]: The dictionary representation of the node.
        """

        # kind is necessary for all nodes.
        graph = {"kind": self.kind}

        if "tensor" in self.kind or "scalar" in self.kind:
            graph.update(
                {
                    "index": self.index,
                    "shape": self.shape,
                    "dtype": self.dtype,
                    "x": self.x if self.store_forward_pass_output else None,
                }
            )
        else:
            graph.update(
                {
                    # Note: key 'op' was changed to 'op_name' to be clarified its purpose.
                    "op_name": self.op_name,
                    "operands": [operand.to_dict() for operand in self.operands] if self.operands else None,
                    "args": self.args,
                    "kwargs": self.kwargs,
                }
            )

        return graph

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_NodeForTracingMode":
        """
        Create a `_NodeForTraceMode` instance from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary to create the instance from.

        Returns:
            _NodeForTracingMode: The created instance.
        """
        kind = data.get('kind')
        if 'tensor' in kind or "scalar" in kind:
            node = cls(
                kind=kind,
                store_forward_pass_output=data.get('store_forward_pass_output', False),
                allow_conditional_flow=data.get('allow_conditional_flow', True),
                x=data.get('x', None),
                shape=data.get('shape', None),
                dtype=data.get('dtype', None),
            )
            node.index = data.get('index', None)
            return node

        else:
            operands = [
                cls.from_dict(op)
                for op in data.get('operands', [])
            ]
            return cls(
                kind=kind,
                store_forward_pass_output=data.get('store_forward_pass_output', False),
                allow_conditional_flow=data.get('allow_conditional_flow', True),
                x=data.get('x', None),
                op_name=data.get('op_name', None),
                operands=operands,
                args=data.get('args', None),
                kwargs=data.get('kwargs', None),
            )

    def __repr__(self):
        """Return a string representation of the node."""
        return f"_NodeForTraceMode(kind={self.kind}, op_name={self.op_name}, operands={self.operands}, " \
               f"shape={self.shape}, dtype={self.dtype}, args={self.args}, kwargs={self.kwargs})"

    def __str__(self):
        """Return a string representation of the node."""
        return f"_NodeForTraceMode(kind={self.kind}, op_name={self.op_name}, operands={self.operands}, " \
               f"shape={self.shape}, dtype={self.dtype}, args={self.args}, kwargs={self.kwargs})"

    def shape(self):
        """Return the shape of the input tensor."""
        return self.shape


def _create_methods(op_info: Dict[str, Any], op_name: str) -> Tuple[Callable, Callable]:
    """
    Create the eager and tracing methods dynamically from the pre-defined operation map.
    Since there are so many operations, this function was introduced to avoid code duplication.

    Args:
        op_info (Dict[str, Any]): The operation information from the JVP_MAP.
        op_name (str): The operation name.

    Returns:
        Tuple[Callable, Callable]: The eager and trace methods.
    """
    if op_info["kind"] == "cond":
        def eager_method(self, *args, **kwargs):
            return op_info["fwd"](self, *args, **kwargs)

        def trace_method(self, *args, **kwargs):
            if self.allow_conditional_flow:
                return op_info["fwd"](self, *args, **kwargs)
            else:
                raise ValueError(
                    "Conditional operations are not allowed in the tracing mode. "
                    "To allow them, set `allow_conditional_flow` to `True`."
                )

    elif op_info["kind"] == "binary":
        def eager_method(self, other, *args, **kwargs):
            if not isinstance(other, _NodeForEagerMode):
                other = np.array(other)
                other = _NodeForEagerMode(other, np.zeros_like(other))
            return _NodeForEagerMode(
                op_info["fwd"](self, other, *args, **kwargs),
                op_info["bwd"](self, other, *args, **kwargs) if self.need_bwd else None,
            )

        def trace_method(self, other, *args, **kwargs):
            if not isinstance(other, _NodeForTracingMode):
                other = np.array(other)
                other = _NodeForTracingMode(
                    kind="python-scalar",
                    allow_conditional_flow=self.allow_conditional_flow,
                    store_forward_pass_output=self.store_forward_pass_output,
                    x=other,
                    shape=other.shape,
                    dtype=other.dtype.name,
                    args=args,
                    kwargs=kwargs,
                )
            return _NodeForTracingMode(
                kind=op_info["kind"],
                allow_conditional_flow=self.allow_conditional_flow,
                store_forward_pass_output=self.store_forward_pass_output,
                x=op_info["fwd"](self, other, *args, **kwargs) if self.need_fwd else None,
                op_name=op_name,
                operands=[self, other],
                args=args,
                kwargs=kwargs,
            )
    else:
        def eager_method(self, *args, **kwargs):
            return _NodeForEagerMode(
                op_info["fwd"](self, *args, **kwargs),
                op_info["bwd"](self, *args, **kwargs) if self.need_bwd else None,
            )

        def trace_method(self, *args, **kwargs):
            return _NodeForTracingMode(
                kind=op_info["kind"],
                allow_conditional_flow=self.allow_conditional_flow,
                store_forward_pass_output=self.store_forward_pass_output,
                x=op_info["fwd"](self, *args, **kwargs) if self.need_fwd else None,
                op_name=op_name,
                operands=[self],
                args=args,
                kwargs=kwargs,
            )

    return eager_method, trace_method


def _clear_tensor_index():
    """Clear the `_TENSOR_INDEX` counter."""
    global _TENSOR_INDEX
    _TENSOR_INIT_LOCK.acquire()
    _TENSOR_INDEX = 0
    _TENSOR_INIT_LOCK.release()


# Dynamically create the eager and tracing methods for each operation
# and assign them to the `JVPNodeForEagerMode` and `JVPNodeForTraceMode` classes.
for _op_name, _op_info in _JVP_MAP.items():
    _eager_method, _trace_method = _create_methods(_op_info, _op_name)
    if "property" in _op_info and _op_info["property"] is True:
        # This was added because of the `np.ndarray.T` property.
        setattr(_NodeForEagerMode, _op_name, property(_eager_method))
        setattr(_NodeForTracingMode, _op_name, property(_trace_method))
    else:
        setattr(_NodeForEagerMode, _op_name, _eager_method)
        setattr(_NodeForTracingMode, _op_name, _trace_method)

        if "magic" in _op_info:
            for _magic_name in _op_info["magic"]:
                setattr(_NodeForEagerMode, _magic_name, _eager_method)
                setattr(_NodeForTracingMode, _magic_name, _trace_method)


def jvp(
    func: Callable,
    inputs: Tuple[np.ndarray, ...],
    grads: Optional[Tuple[np.ndarray, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Jacobian-vector product of a function with given inputs and grads.

    Args:
        func (Callable): The function to compute the Jacobian-vector product of.
        inputs (Tuple[np.ndarray, ...]): The inputs to the function.
        grads (Tuple[np.ndarray, ...]): The grads to be producted with the Jacobian.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            The output of the function and the result of the Jacobian-vector product.
    """

    if len(inputs) != len(grads):
        raise ValueError("The number of `inputs` and the number of `grads` must match.")

    if isinstance(inputs, np.ndarray):
        inputs = (inputs,)

    if not isinstance(inputs[0], np.ndarray):
        raise ValueError("The `inputs` must be numpy arrays.")

    if isinstance(grads, np.ndarray):
        grads = (grads,)

    if not isinstance(grads[0], np.ndarray):
        raise ValueError("The `grads` must be numpy arrays.")

    outputs = func(*[_NodeForEagerMode(i, g) for i, g in zip(inputs, grads)])

    return outputs.x, outputs.g


def custom_fn(func: Callable):
    """
    Decorator to wrap a function to work with eager mode.

    Args:
        func (Callable): The function to be wrapped.

    Return:
        Callable: The wrapped function.
    """
    def wrapper(*inputs, **kwargs):
        """
        Args:
            inputs (Tuple[np.ndarray, ...]): The inputs to the function.
            kwargs (Dict[str, Any]): The keyword arguments to the function.

        Return:
            np.ndarray: The output of the function.
        """
        return func(*[_NodeForEagerMode(x) for x in inputs], **kwargs).x

    return wrapper
