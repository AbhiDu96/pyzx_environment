import numpy as np
from zx_env.env import ZXEnv


if __name__ == '__main__':

    # normalized_cnot_count_reward matches the paper's objective (two-qubit gate
    # reduction) and returns the (reward, level) pair that the environment expects.
    env = ZXEnv(n_qubits=5, depth=250, max_steps=100,
                reward_fn="normalized_cnot_count_reward")

    obs, info = env.reset()
    state, action_masks, zx_graph, node_masks, edge_masks, rule_mask = obs
    print(f"reset ok: {env.action_space.n} rules, "
          f"graph with {zx_graph.num_vertices()} vertices / {zx_graph.num_edges()} edges")
    print(f"  initial reward={info['reward']:.4f}, feats shape={tuple(info['feats'].shape)}")

    for step in range(5):
        # pick a random legal (rule, position) pair from the action masks
        legal = np.argwhere(action_masks[:, 1:] == 1)
        if len(legal) == 0:
            print("  no legal actions remaining")
            break
        rule, position = legal[np.random.randint(len(legal))]
        obs, reward, terminated, truncated, info = env.step(action=int(rule), position=int(position))
        action_masks = obs[1]
        print(f"  step {step}: rule={env.rules_list[int(rule)]:<20} "
              f"reward={reward:.4f} terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            break

    print("SMOKE TEST PASSED")
