"""Load small structured circuits from MQTBench as PyZX circuits/graphs.

Pipeline: mqt.bench (qiskit circuit) -> transpile to a pyzx-friendly basis -> QASM 2
-> ``pyzx.Circuit.from_qasm``. Generated QASM strings are cached on disk (JSON) so
repeated runs do not regenerate the benchmarks.

``mqt-bench`` and ``qiskit`` are only imported when a pool actually has to be
generated, so the rest of the package works without them.
"""

from __future__ import annotations

import json
import logging
import os
import re
from fractions import Fraction
from dataclasses import dataclass

import numpy as np
import pyzx as zx

# Gates understood by pyzx's QASM parser that we let qiskit emit.
PYZX_BASIS_GATES = ["rz", "rx", "h", "cx", "cz", "x", "z", "s", "t", "sdg", "tdg"]
_CACHE_VERSION = 2
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "zx_env", "mqtbench")


@dataclass
class MQTBenchEntry:
    name: str
    n_qubits: int
    n_gates: int
    qasm: str

    def to_circuit(self) -> zx.Circuit:
        return zx.Circuit.from_qasm(self.qasm)


def _qiskit_to_qasm(qc) -> str:
    """Transpile a qiskit circuit to the pyzx basis and dump QASM 2."""
    import qiskit
    from qiskit import qasm2

    qc = qc.remove_final_measurements(inplace=False)
    qc = qiskit.transpile(qc, basis_gates=PYZX_BASIS_GATES, optimization_level=1, seed_transpiler=0)
    return qasm2.dumps(qc)


def _count_gates(qasm: str) -> int:
    return sum(
        1
        for line in qasm.splitlines()
        if line.strip() and not re.match(r"\s*(OPENQASM|include|qreg|creg|barrier|//)", line)
    )


def _is_clifford_t(circ: zx.Circuit) -> bool:
    """True if every gate phase is an integer multiple of pi/4 (required by the env)."""
    return all((Fraction(getattr(g, "phase", 0)) * 4).denominator == 1 for g in circ.gates)


def _cache_path(cache_dir, qubit_counts, level, benchmarks) -> str:
    key = f"v{_CACHE_VERSION}_{level}_q{min(qubit_counts)}-{max(qubit_counts)}"
    if benchmarks is not None:
        key += "_" + "-".join(sorted(benchmarks))
    return os.path.join(cache_dir, key + ".json")


def generate_entries(qubit_counts=(2, 3, 4, 5), max_gates=100, level="indep", benchmarks=None,
                     seed=0) -> list[MQTBenchEntry]:
    """Generate (uncached) MQTBench entries with <= max_gates gates after transpilation."""
    from mqt.bench import BenchmarkLevel, get_benchmark
    from mqt.bench.benchmarks import get_available_benchmark_names

    lvl = {"alg": BenchmarkLevel.ALG, "indep": BenchmarkLevel.INDEP}[level]
    names = list(benchmarks) if benchmarks is not None else get_available_benchmark_names()
    entries: list[MQTBenchEntry] = []
    for name in names:
        for n in qubit_counts:
            try:
                np.random.seed(seed)
                qasm = _qiskit_to_qasm(get_benchmark(name, lvl, n))
                circ = zx.Circuit.from_qasm(qasm)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # unsupported size/gates; also rust PanicException (BaseException)
                logging.debug("mqtbench %s n=%d skipped: %s", name, n, exc)  # noqa: LOG015
                continue
            n_gates = len(circ.gates)
            if circ.qubits != n or n_gates == 0:
                continue
            if not _is_clifford_t(circ):  # env's T-count code requires multiples of pi/4
                continue
            entries.append(MQTBenchEntry(name, circ.qubits, n_gates, qasm))
    return entries


def load_entries(qubit_counts=(2, 3, 4, 5), max_gates=100, level="indep", benchmarks=None, seed=0,
                 cache_dir=None, use_cache=True) -> list[MQTBenchEntry]:
    """Return MQTBench entries, using an on-disk JSON cache when available."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    path = _cache_path(cache_dir, qubit_counts, level, benchmarks)
    if use_cache and os.path.exists(path):
        with open(path) as f:
            entries = [MQTBenchEntry(**e) for e in json.load(f)]
    else:
        entries = generate_entries(qubit_counts, 10**9, level, benchmarks, seed)  # cache unfiltered by gates
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump([e.__dict__ for e in entries], f)
    # circuits without any two-qubit gate have nothing to optimise (and may have no legal actions)
    return [e for e in entries if e.n_gates <= max_gates and e.n_qubits in qubit_counts
            and re.search(r"^\s*(cx|cz)\b", e.qasm, re.M)]


class MQTBenchPool:
    """A pool of structured circuits that can be sampled (optionally by qubit count)."""

    def __init__(self, entries: list[MQTBenchEntry]):
        if not entries:
            raise ValueError("MQTBench pool is empty")
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def counts_per_qubit(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for e in self.entries:
            out[e.n_qubits] = out.get(e.n_qubits, 0) + 1
        return dict(sorted(out.items()))

    def sample_entry(self, n_qubits: int | None = None, rng=None) -> MQTBenchEntry:
        rand = rng if rng is not None else np.random
        cands = self.entries if n_qubits is None else [e for e in self.entries if e.n_qubits == n_qubits]
        if not cands:  # fall back to the whole pool if no circuit has this qubit count
            cands = self.entries
        return cands[int(rand.integers(len(cands)) if hasattr(rand, "integers") else rand.randint(len(cands)))]

    def sample_circuit(self, n_qubits: int | None = None, rng=None) -> zx.Circuit:
        return self.sample_entry(n_qubits, rng).to_circuit()

    def sample_graph(self, n_qubits: int | None = None, rng=None):
        return self.sample_circuit(n_qubits, rng).to_graph()


def build_pool(qubit_counts=(2, 3, 4, 5), max_gates=100, level="indep", benchmarks=None, seed=0,
               cache_dir=None, use_cache=True) -> MQTBenchPool:
    """Build a pool of MQTBench circuits (2..5 qubits, <= max_gates gates by default)."""
    return MQTBenchPool(load_entries(tuple(qubit_counts), max_gates, level, benchmarks, seed, cache_dir, use_cache))


def build_graph_pool(**kwargs) -> list:
    """Convenience: list of pyzx graphs, one per pool circuit."""
    return [e.to_circuit().to_graph() for e in build_pool(**kwargs).entries]
