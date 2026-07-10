# pyzx_environment (`zx_env`)

A [Gymnasium](https://gymnasium.farama.org/) reinforcement-learning environment for
optimizing quantum circuits by rewriting their [ZX diagrams](https://pyzx.readthedocs.io/).
The environment turns a quantum circuit into a ZX graph, exposes the graph (and a set of
hand-crafted features) as an observation, and accepts actions that correspond to
[PyZX](https://github.com/Quantomatic/pyzx) graph-rewriting rules applied at a chosen
node or edge.

It is the environment package used by the paper **"Optimizing Quantum Circuits via ZX
Diagrams using Reinforcement Learning and Graph Neural Networks"**
(agent/training code: <https://github.com/AbhiDu96/ZXDiagramSimplification>). This
repo contains only the environment; the RL agents, tree search, and training loops live in
the parent `ZXDiagramSimplification/` project, which installs this package as a dependency.

The optimization target is reducing the number of **two-qubit gates (CNOT/CZ)** — and,
depending on the reward function, the **T-count** — of the extracted circuit, while keeping
it semantically equivalent to the original.

## Installation

Requires Python ≥ 3.8. Install the package (editable) from this directory:

```bash
pip install -e .
```

Core runtime dependencies (installed separately, see the parent project for pinned
versions):

```bash
pip install pyzx gymnasium numpy torch torch-geometric matplotlib
```

## Quick start

```python
import numpy as np
from zx_env import zx_env   # class is re-exported from zx_env.env

# Create an environment that samples random 5-qubit, depth-250 Clifford+T circuits
env = zx_env(n_qubits=5, depth=250, max_steps=100)

obs, info = env.reset()
state, action_masks, zx_graph, node_masks, edge_masks, rule_mask = obs

# Pick a random *legal* (rule, position) pair from the action masks and step
rule = np.random.randint(env.action_space.n)          # which rewrite rule
legal_positions = np.flatnonzero(action_masks[rule, 1:])  # applicable node/edge slots
position = int(np.random.choice(legal_positions))
obs, reward, terminated, truncated, info = env.step(action=rule, position=position)
```

See [`test.py`](test.py) for a minimal end-to-end smoke test.

## Package layout

```
pyzx_environment/
├── setup.py                              # installs the `zx_env` package
├── test.py                               # minimal usage / smoke test
└── zx_env/
    ├── env.py                            # zx_env: the Gymnasium environment (main entry point)
    ├── rules/
    │   └── custom_rules.py               # ZX rewrite-rule matchers + rewriters (the action space)
    ├── circuit_utils/
    │   ├── circuit_generator.py          # random_circuit(): Clifford+T circuit sampler
    │   ├── circuit_extractor.py          # multi-level ZX-graph → circuit extraction (levels 1–5)
    │   ├── graph_format_converter.py                 # pyzx graph → torch_geometric conversion
    │   └── graph_format_converter_index_adjusted.py  # index-remapped variant used by the env
    ├── general_utils/
    │   ├── reward_functions.py           # reward definitions (T-count / CNOT-count based)
    │   └── utils.py                      # check_equality(), tcount_from_graph()
    └── bench_mark_circuits/              # Quipper benchmark circuits (qubit_3, qubit_4, qubit_5)
```

## How the environment works

Each episode proceeds roughly as follows:

1. **Sample a circuit.** `reset()` draws a random Clifford+T circuit
   (`circuit_utils/circuit_generator.py`) sized by `n_qubits` / `depth` and the gate-mix
   ratios (`mq_ratio`, `h_ratio`, `t_ratio`), then converts it to a ZX graph. Baselines are
   recorded: the original T-count / CNOT-count, and the counts after PyZX's `full_reduce`
   (used as reference points by some rewards).
2. **Optional mutation.** With probability `mutate_probability`, the starting graph is
   perturbed by up to `mutation_steps` random rewrites, so the agent does not always start
   from a "clean" circuit.
3. **Observe.** The current ZX graph is converted to a `torch_geometric` graph and paired
   with action masks (see below).
4. **Act.** `step(action, position)` applies rewrite rule `action` at graph location
   `position`. Identity/trivial simplifications may be applied depending on config
   (`full_fuse_every_step`).
5. **Reward & termination.** Reward is computed by the configured reward function. An
   episode ends when no action is applicable, `max_steps` is reached, or a rewrite raises an
   exception. On a *clean* termination the environment verifies semantic equivalence to the
   original circuit and extracts the optimized circuit for its final statistics.

### Action space

`env.action_space` is `Discrete(len(rules_list))` — the index selects **which rewrite
rule** to apply. The **location** at which to apply it is supplied separately as the
`position` argument to `step()`:

- For **node-based** rules, `position` indexes the graph's nodes.
- For **edge-based** rules, `position` indexes the (undirected) edge list, offset by the
  number of nodes (`position - num_nodes`), matching the mask layout below.

The available rules (the `match_*` functions in `rules/custom_rules.py`):

| Rule (`rules_list` name) | Target | Effect |
| --- | --- | --- |
| `spider_fusion`        | edge | Merge two connected same-color spiders, adding their phases. |
| `unspider`             | node | Split a spider, pushing phase onto a new adjacent spider (inverse of fusion). |
| `complete_unfuse`      | node | Unfuse a high-degree spider (degree > 3) into a complete graph of spiders. |
| `pi_copy`              | edge | Push a π-phase (Pauli) spider through an adjacent spider. |
| `color_change`         | node | Toggle a spider's color (Z↔X), adding Hadamards on its edges. |
| `bialgebra`            | edge | Apply the X/Z bialgebra rule across an edge (powerful but graph-expanding). |
| `euler`                | edge | Decompose a Hadamard edge into an Euler sequence of π/2-phase spiders. |
| `add_identity`         | edge | Insert an identity (0-phase Z) spider on an edge. |
| `add_hadamard_identity`| edge | Insert an identity spider with Hadamard edges on a simple edge. |

By default all `match_*` rules are enabled; pass `rules_list=[...]` (names **without** the
`match_` prefix) to restrict the set. `full_fuse` / `full_id_remove` are "hard"
simplification passes used internally (e.g. `full_fuse_every_step`) and are **not** part of
the action space.

### Observation

`reset()` and `step()` return the observation as a 6-element list:

```
[state, action_masks, state_zx_graph, node_masks, edge_masks, rule_mask]
```

| Element | Type | Meaning |
| --- | --- | --- |
| `state`          | `torch_geometric` `Data` / `HeteroData` | The ZX graph encoded for a GNN (made undirected). |
| `action_masks`   | `np.ndarray` `(n_rules, 1 + n_nodes + 2·n_edges)` | Per rule: `[not_possible_flag, node_mask, edge_mask]`. |
| `state_zx_graph` | `pyzx.Graph` | The raw ZX graph (for extraction, plotting, equality checks). |
| `node_masks`     | `np.ndarray` `(n_rules, n_nodes)` | Which nodes each rule can be applied to. |
| `edge_masks`     | `np.ndarray` `(n_rules, 2·n_edges)` | Which edges each rule can be applied to. |
| `rule_mask`      | `np.ndarray` `(n_rules,)` | Whether each rule has *any* legal match. |

The accompanying `info` dict includes:

- `feats` — an 8-dimensional feature vector of the current circuit (see below).
- `action_mask`, `reward`, `level` (the extraction level used, 1–5).
- On clean termination: `applied_rule`, initial/final/full-reduce circuit statistics
  (`*_cnot_count`, `*_t_count`), and the extracted `init_circuit` / `final_circuit`.

If `add_no_action=True`, an extra "no-op" action is appended to the action set.

### Feature vector (`mk_features`)

A compact 8-dim summary of the extracted circuit, each entry normalized by the expected
circuit size (`≈ depth × n_qubits`):

`[gates, tcount, clifford, twoqubit, had, depth, depth_cz, edges]`

### Graph encodings

Set via `graph_type`:

- `"homogeneous"` (default) — a single node type; node features are `[spider_type, phase]`
  and edges carry their edge type. (`pyzy_to_homogeneous_torchData`)
- `"heterogeneous"` — a `HeteroData` graph with distinct `inNodes` / `outNodes` /
  `xSpiders` / `zSpiders` node types and typed edges between them.
  (`pyzx_to_heterogeneous_torchData`)

### Rewards

Selected with `reward_fn` (a string name or a custom callable). Built-in options in
`general_utils/reward_functions.py`:

| `reward_fn` | Definition |
| --- | --- |
| `normalized_t_count_reward`      | `1 − t_count / baseline_t_count` |
| `absolute_t_count_reward`        | `baseline_t_count − t_count` |
| `normalized_cnot_count_reward`   | `1 − cnot_count / baseline_cnot_count` |
| `absolute_cnot_count_reward`     | `baseline_cnot_count − cnot_count` |
| `pyzx_normalized_t_count_reward` | normalized against the PyZX `full_reduce` T-count baseline |
| `pyzx_normalized_cnot_count_reward` | normalized against the PyZX `full_reduce` CNOT baseline |

CNOT-based rewards require extracting a circuit from the current graph; extraction uses the
method chosen by `circuit_extraction_type` (see below).

Special reward values signal failure modes: applying a **masked-out** action yields a large
negative penalty and ends the episode, a rewrite that raises an **exception** yields a
negative penalty and marks the episode "dead", and a final graph that fails the
**semantic-equality** check is heavily penalized.

### Circuit extraction & equivalence checking

- `circuit_utils/circuit_extractor.py` provides a custom, **graph-like** extraction that
  tries increasingly aggressive simplifications (**levels 1–4**, then falls back to PyZX
  `full_reduce` at **level 5**) to keep the two-qubit gate count low. Choose it with
  `circuit_extraction_type="custom"` (default) or use plain PyZX extraction otherwise.
- `general_utils/utils.py:check_equality` verifies that the rewritten graph is still
  equivalent to the original by composing one graph with the adjoint of the other and
  checking that `full_reduce` collapses it to the identity. (A `False` result means the
  reduction *did not succeed*, not necessarily that the circuits differ.)

## Key configuration parameters

Constructor arguments of `zx_env(...)` (see [`zx_env/env.py`](zx_env/env.py) for the full
list and defaults):

| Parameter | Default | Description |
| --- | --- | --- |
| `n_qubits` | `5` | Number of qubits; an `int`, or a `(low, high)` range to sample from. |
| `depth` | `250` | Number of gates in the sampled circuit (`int` or range). |
| `max_steps` | `100` | Maximum rewrite steps per episode. |
| `mq_ratio`, `h_ratio`, `t_ratio` | `0.1, 0.3, 0.5` | Two-qubit / Hadamard / T-rotation gate mix of sampled circuits. |
| `rules_list` | `None` (all rules) | Restrict the action set (names without the `match_` prefix). |
| `graph_type` | `"homogeneous"` | Graph encoding for the observation (`"homogeneous"` / `"heterogeneous"`). |
| `reward_fn` | `"normalized_t_count_reward"` | Reward function (name or callable). |
| `circuit_extraction_type` | `"custom"` | `"custom"` multi-level extractor vs. plain PyZX extraction. |
| `mutate_graph`, `mutate_probability`, `mutation_steps` | `True, 0.5, 100` | Random perturbation of the starting graph. |
| `min_t_count_diff` | `0.1` | Threshold controlling which sampled circuits are accepted at reset. |
| `add_no_action` | `False` | Add an explicit no-op action. |
| `full_fuse_every_step` | `False` | Run `spider_simp` fusion after every step. |
| `reduce_at_reset` | `False` | Fuse spiders once when resetting. |
| `negative_reward_mean`, `negative_reward_std` | `-0.1, 0.0` | Reward used for neutral/no-progress terminations. |

## Benchmark circuits

`zx_env/bench_mark_circuits/` holds standard Quipper-format circuits (`qubit_3`,
`qubit_4`, `qubit_5` — e.g. `tof_3`, `barenco_tof_4`, `gf2^5_mult`, `mod5_4`). They are
loaded automatically at construction (`bench_mark()`) and are useful for evaluating a
trained agent on fixed, well-known circuits rather than random samples.
