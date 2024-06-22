#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains code about graph-based forward-mode automatic differentiation.
"""

import ast
import inspect
from typing import Union, List, Tuple, Dict, Any, Callable, Optional

import numpy as np

from mygrad.core import (
    _JVP_MAP,
    _PYTHON_SCALR_MAP,
    _NodeForEagerMode,
    _NodeForTracingMode,
    _clear_tensor_index,
)


def _extract_args(
    args: Union[np.ndarray, List, Tuple, Dict],
    allow_conditional_flow: bool,
    store_forward_pass_output: bool,
) -> Union[_NodeForTracingMode, np.ndarray, List, Tuple, Dict]:
    """
    Extract the arguments from the inputs and return the corresponding
    `_JVPNodeForTraceMode` instances. This extracts all sub-arguments recursively.

    Args:
        args (Union[np.ndarray, List, Tuple, Dict]): The arguments to be extracted.
        allow_conditional_flow (Union[str, bool]): Whether to allow conditional operations.
            If "auto", it will be determined based on the function. Defaults to "auto".
            If it's True, forward pass will be done to determine the conditional flow.
        store_forward_pass_output (bool): Whether to store the forward pass output in the graph.
            It can be useful to check which input tensors are used in the forward pass when debugging the graph.
            Use `dill` instead of `json` to serialize the graph when this option is enabled.

    Returns:
        Union[_NodeForTracingMode, np.ndarray, List, Tuple, Dict]:
            The extracted arguments.
    """
    if isinstance(args, (list, tuple)):
        return [
            _extract_args(
                arg,
                allow_conditional_flow=allow_conditional_flow,
                store_forward_pass_output=store_forward_pass_output,
            )
            for arg in args
        ]
    elif isinstance(args, dict):
        return {
            key: _extract_args(
                value,
                allow_conditional_flow=allow_conditional_flow,
                store_forward_pass_output=store_forward_pass_output,
            )
            for key, value in args.items()
        }
    elif isinstance(args, _NodeForTracingMode):
        return args
    elif isinstance(args, np.ndarray):
        # Add real value to tracing node because we need to use it for conditional flow.
        # We can evaluate the operation directly and add correct path to the graph.
        # Note that current implementation only supports static conditional flow.
        return _NodeForTracingMode(
            kind="tensor",
            x=args,
            shape=args.shape,
            dtype=args.dtype.name,
            allow_conditional_flow=allow_conditional_flow,
            store_forward_pass_output=store_forward_pass_output,
        )
    else:
        raise RuntimeError("Arguments must be tensors.")


def _flatten_args(args: Any) -> Any:
    """
    Flatten the arguments recursively.

    Args:
        args (Any): The arguments to be flattened.

    Yields:
        Any: The flattened arguments.
    """
    if isinstance(args, (list, tuple)):
        for arg in args:
            yield from _flatten_args(arg)
    elif isinstance(args, dict):
        for _, value in args.items():
            yield from _flatten_args(value)
    else:
        yield args


def _execute_recursively(
    graph: Union[Dict[str, Any], _NodeForTracingMode],
    inputs: Tuple[np.ndarray, ...],
    grads: Optional[Tuple[np.ndarray, ...]] = None,
) -> _NodeForEagerMode:
    """
    Execute the graph and return the last eager mode node.
    This function is called recursively to execute all the sub-graphs.

    Args:
        graph (Union[Dict[str, Any], _NodeForTracingMode]): The graph to be executed.
        inputs (Tuple[np.ndarray, ...]): The inputs for the function.
        grads (Optional[Tuple[np.ndarray, ...]]): The grads for the inputs.

    Returns:
        _NodeForEagerMode: The last eager mode node.
    """
    if isinstance(graph, dict):
        graph = _NodeForTracingMode.from_dict(graph)

    inputs = dict(enumerate(_flatten_args(inputs)))
    if grads is not None:
        grads = dict(enumerate(_flatten_args(grads)))

    if isinstance(graph, _NodeForTracingMode):
        if graph.kind == "tensor":
            output = _NodeForEagerMode(
                x=inputs[graph.index],
                g=None if grads is None else grads[graph.index],
            )
        elif graph.kind == "python-scalar":
            x = _PYTHON_SCALR_MAP[graph.index]
            output = _NodeForEagerMode(
                x=x,
                g=None if grads is None else np.zeros_like(x),
            )
        elif graph.kind == "binary":
            left = _execute_recursively(graph.operands[0], inputs, grads)
            right = _execute_recursively(graph.operands[1], inputs, grads)
            args = graph.args if graph.args is not None else ()
            kwargs = graph.kwargs if graph.kwargs is not None else {}
            output = _NodeForEagerMode(
                # Skip forward pass if the forward pass output is already stored in the graph.
                x=_JVP_MAP[graph.op_name]["fwd"](left, right, *args, **kwargs) if graph.x is None else graph.x,
                g=_JVP_MAP[graph.op_name]["bwd"](left, right, *args, **kwargs) if grads is not None else None,
            )
        elif graph.kind in ["unary", "elementwise", "reduce"]:
            output = _execute_recursively(graph.operands[0], inputs, grads)
            args = graph.args if graph.args is not None else ()
            kwargs = graph.kwargs if graph.kwargs is not None else {}
            output = _NodeForEagerMode(
                # Skip forward pass if the forward pass output is already stored in the graph.
                x=_JVP_MAP[graph.op_name]["fwd"](output, *args, **kwargs) if graph.x is None else graph.x,
                g=_JVP_MAP[graph.op_name]["bwd"](output, *args, **kwargs) if grads is not None else None,
            )
        else:
            # conditional operations are not stored in the graph.
            # They are determined during the tracing phase.
            raise ValueError(
                f"'{graph.kind}' is not supported kind. "
                f"The supported kinds are 'tensor', 'python-scalar', "
                f"'binary', 'unary', 'elementwise', and 'reduce'."
            )
        return output
    else:
        raise ValueError(
            "Parameter `graph` must be a dictionary or a JVPNodeForTraceMode instance."
        )


def _has_conditional_flow(func: Callable) -> bool:
    """
    Check if a Python function contains conditional statements (if).

    Args:
        func (Callable): The function to check.

    Returns:
        bool: True if the function contains conditional statements, False otherwise.
    """
    # Get the source code of the function
    source = inspect.getsource(func)
    lines = source.splitlines()
    min_indent = min((len(line) - len(line.lstrip())) for line in lines if line.lstrip())

    # Remove the common leading whitespace from each line
    cleaned_source = "\n".join(line[min_indent:] for line in lines)

    # Parse the cleaned source code into an AST
    tree = ast.parse(cleaned_source)

    # Define a visitor class to traverse the AST
    class ConditionalFlowVisitor(ast.NodeVisitor):
        def __init__(self):
            self.has_conditional_flow = False

        def visit_If(self, node):
            self.has_conditional_flow = True
            self.generic_visit(node)

    # Create an instance of the visitor and visit the AST
    visitor = ConditionalFlowVisitor()
    visitor.visit(tree)

    # Return whether conditional flow statements were found
    return visitor.has_conditional_flow


def trace(
    func: Callable,
    *args: Union[np.ndarray, List, Tuple, Dict],
    allow_conditional_flow: Union[str, bool] = "auto",
    store_forward_pass_output: bool = False,
) -> Dict[str, Any]:
    """
    Trace the function and return the corresponding graph.

    Args:
        func (callable): The function to be traced.
        *args (Union[np.ndarray, List, Tuple, Dict]): The arguments to used for tracing.
        allow_conditional_flow (Union[str, bool]): Whether to allow conditional operations.
            If "auto", it will be determined based on the function. Defaults to "auto".
            If it's True, forward pass will be done to determine the conditional flow.
        store_forward_pass_output (bool): Whether to store the forward pass output in the graph.
            It can be useful to check which input tensors are used in the forward pass when debugging the graph.
            Use `dill` instead of `json` to serialize the graph when this option is enabled.

    Returns:
        Dict[str, Any]: The graph traced for the function.
    """
    _clear_tensor_index()

    assert allow_conditional_flow in [True, False, "auto"]

    if allow_conditional_flow == "auto":
        allow_conditional_flow = _has_conditional_flow(func)

    outputs = func(
        *_extract_args(
            args,
            allow_conditional_flow=allow_conditional_flow,
            store_forward_pass_output=store_forward_pass_output,
        )
    )

    return outputs.to_dict()


def execute(
    graph: Union[Dict[str, Any], _NodeForTracingMode],
    inputs: Tuple[np.ndarray, ...],
    grads: Optional[Tuple[np.ndarray, ...]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Execute the function and return the result.

    Args:
        graph (Union[Dict[str, Any], _NodeForTracingMode]): The graph to be executed.
        inputs (Tuple[np.ndarray, ...]): The inputs for the function.
        grads (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): The grads for the inputs.

    Returns:
        Tuple[np.ndarray, ...]: The output of the forward pass and its gradient.
    """
    output = _execute_recursively(graph, inputs, grads)
    return output.x if grads is None else (output.x, output.g)
