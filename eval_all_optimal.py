# frozenlake_sf_eval_all_goals.py
import gymnasium as gym
import torch
import os

# -------- config (must match training) --------
env_id = "FrozenLake-v1"
is_slippery = False
max_steps_per_episode = 100
num_eval_episodes = 100

# ---- custom map with NO built-in G (same as training) ----
custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFF",
]

phi_folder = "phi"   # where psi tables (saved as "phi") live
w_folder = "w"       # where w vectors live


class TaskRewardWrapper(gym.Wrapper):
    """
    Same as in training:
      - reward = 1 when the agent is in goal_state
      - reward = 0 otherwise
    Does NOT terminate on reaching the goal; episodes end only due to:
      - falling into a hole (base env terminal), or
      - time limit (truncation wrapper in Gymnasium).
    """
    def __init__(self, env, goal_state: int):
        super().__init__(env)
        self.goal_state = goal_state

    def step(self, action):
        obs, r_env, terminated_env, truncated_env, info = self.env.step(action)
        s = int(obs) if not isinstance(obs, (tuple, list)) else int(obs[0])

        reward = 1.0 if s == self.goal_state else 0.0

        terminated = terminated_env
        truncated = truncated_env
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env(goal_state: int):
    base_env = gym.make(
        env_id,
        desc=custom_map,
        is_slippery=is_slippery,
        render_mode=None,  # no video during eval
    )
    env = TaskRewardWrapper(base_env, goal_state=goal_state)
    return env


def get_available_goal_states(phi_folder, w_folder):
    """
    Look at phi/ and w/ folders and return sorted list of goal IDs
    that have BOTH a phi and a w file: <goal>.pth in each.
    """
    phi_goals = set()
    w_goals = set()

    if os.path.isdir(phi_folder):
        for fname in os.listdir(phi_folder):
            if fname.endswith(".pth"):
                stem = os.path.splitext(fname)[0]
                if stem.isdigit():
                    phi_goals.add(int(stem))

    if os.path.isdir(w_folder):
        for fname in os.listdir(w_folder):
            if fname.endswith(".pth"):
                stem = os.path.splitext(fname)[0]
                if stem.isdigit():
                    w_goals.add(int(stem))

    goals = sorted(phi_goals & w_goals)
    return goals


def eval_for_goal(goal_state: int):
    # ---- load learned SF table (psi) and w for this goal ----
    psi_path = os.path.join(phi_folder, f"{goal_state}.pth")
    w_path = os.path.join(w_folder, f"{goal_state}.pth")

    print(f"\n===== Evaluating goal_state = {goal_state} =====")
    print(f"Loading psi from: {psi_path}")
    print(f"Loading w   from: {w_path}")

    psi = torch.load(psi_path)   # shape: (S, A, d)
    w = torch.load(w_path)       # shape: (d,)

    env = make_env(goal_state)
    S = env.observation_space.n
    A = env.action_space.n

    # Basic sanity checks
    assert psi.shape[0] == S, f"psi S mismatch: {psi.shape[0]} vs {S}"
    assert psi.shape[1] == A, f"psi A mismatch: {psi.shape[1]} vs {A}"

    returns = []
    steps_per_ep = []
    goal_hits = 0
    hole_terminations = 0

    for ep in range(1, num_eval_episodes + 1):
        obs, info = env.reset()
        s = int(obs) if not isinstance(obs, (tuple, list)) else int(obs[0])

        ep_return = 0.0
        steps = 0
        hit_goal_this_ep = False

        for t in range(max_steps_per_episode):
            # greedy action w.r.t. Q(s,a) = psi[s,a]^T w
            q_vals = torch.matmul(psi[s], w)  # shape: (A,)
            a = int(torch.argmax(q_vals).item())

            step_result = env.step(a)
            if len(step_result) == 5:
                s_dash_raw, r, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                s_dash_raw, r, done, info = step_result
                # if this branch happens, we don't know terminated vs truncated;
                # treat any done as "terminated" for the hole counter
                terminated = done

            s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

            ep_return += float(r)
            steps += 1

            # track if we ever hit this goal state in this episode
            if s_dash == goal_state:
                hit_goal_this_ep = True

            # If base env terminated, it means we hit a hole
            # (since we removed G from the map)
            if done:
                if terminated and s_dash != goal_state:
                    hole_terminations += 1
                break

            s = s_dash

        returns.append(ep_return)
        steps_per_ep.append(steps)
        if hit_goal_this_ep:
            goal_hits += 1

        print(
            f"Goal {goal_state:2d} | Eval Ep {ep:3d} "
            f"| Return={ep_return:.1f} | Steps={steps:3d} | HitGoal={hit_goal_this_ep}"
        )

    avg_return = sum(returns) / len(returns)
    avg_steps = sum(steps_per_ep) / len(steps_per_ep)
    goal_rate = goal_hits / num_eval_episodes

    print("\n----- Evaluation Summary (Goal =", goal_state, ") -----")
    print(f"Eval episodes          : {num_eval_episodes}")
    print(f"Average return         : {avg_return:.3f}")
    print(f"Average steps/episode  : {avg_steps:.2f}")
    print(f"Fraction episodes hit goal_state={goal_state}: {goal_rate:.3f}")
    print(f"Hole terminations      : {hole_terminations}")

    env.close()


def main():
    goals = get_available_goal_states(phi_folder, w_folder)
    if not goals:
        print("No matching phi/<goal>.pth and w/<goal>.pth files found.")
        return

    print("Found goals with psi & w:", goals)

    for g in goals:
        eval_for_goal(g)

    print("\nAll goals evaluated.")


if __name__ == "__main__":
    main()
