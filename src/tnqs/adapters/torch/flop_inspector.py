"""Executed-operation FLOP audit for PyTorch ansatz runtimes."""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from tnqs.domain.costs import RuntimeFlopCounts
from tnqs.domain.specifications import LogPsiOperation


class _OperatorCallCounter(TorchDispatchMode):
    """Record dispatcher calls so unsupported operations remain visible in reports."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: Counter[str] = Counter()
        self.packet_calls: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func)] += 1
        self.packet_calls[str(getattr(func, "_overloadpacket", func))] += 1
        return func(*args, **(kwargs or {}))


class TorchFlopInspector:
    """Measure supported FLOPs while executing one representative ``log_psi`` call.

    This is a dispatcher-level audit, not a hardware performance-counter reading. It therefore
    reports unsupported operator calls rather
    than silently presenting partial coverage as a complete physical instruction count.
    """

    def inspect(self, runtime: object, operation: LogPsiOperation) -> RuntimeFlopCounts:
        wavefunction = runtime.wavefunction
        parameter = next(wavefunction.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        parameter_dtype = (
            parameter.dtype if parameter is not None else torch.get_default_dtype()
        )
        specification = runtime.specification
        lattice = specification.wavefunction.lattice

        configurations = torch.zeros(
            operation.batch_size,
            lattice.Lx,
            lattice.Ly,
            dtype=torch.long,
            device=device,
        )
        was_training = wavefunction.training
        wavefunction.eval()
        flop_counter = FlopCounterMode(display=False)
        call_counter = _OperatorCallCounter()
        try:
            with torch.no_grad(), call_counter, flop_counter:
                wavefunction.log_psi(configurations, None)
        finally:
            wavefunction.train(was_training)

        all_flops = flop_counter.get_flop_counts()
        global_flops = all_flops.get("Global", {})
        by_operator = {
            str(operator): int(count)
            for operator, count in sorted(
                global_flops.items(), key=lambda item: str(item[0])
            )
            if count
        }
        unsupported = {
            name: count
            for name, count in sorted(call_counter.packet_calls.items())
            if name not in by_operator
        }
        supported = {
            name: count
            for name, count in sorted(call_counter.packet_calls.items())
            if name in by_operator
        }
        return RuntimeFlopCounts(
            all_flops=all_flops,
            total=int(flop_counter.get_total_flops()),
            by_operator=by_operator,
            operator_calls=dict(sorted(call_counter.calls.items())),
            unsupported_operator_calls=unsupported,
            supported_operator_calls=supported,
            batch_size=operation.batch_size,
            device=str(device),
            dtype=str(parameter_dtype).removeprefix("torch."),
            include_conditioner=False,
        )


__all__ = ["TorchFlopInspector"]
