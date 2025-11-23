# frozenlake_all_goals_sf.py
import gymnasium as gym
import torch
import os
from tqdm import tqdm
import random

# -------- config --------
env_id = "FrozenLake-v1"
is_slippery = False
num_episodes = 2000
max_steps_per_episode = 100
gamma = 0.99
alpha_psi = 0.15
alpha_w = 0.05

# epsilon schedule
epsilon_start = 1.0
epsilon_end = 0.01

# ---- custom map with NO built-in G ----
custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFF",
]
# Holes in our custom map: 5, 7, 11, 12
# All non-hole tiles (S or F) will be used as possible goal states.

class TaskRewardWrapper(gym.Wrapper):
    """
    Wraps FrozenLake to define our own reward:
      - reward = 1 when the agent is in goal_state
      - reward = 0 otherwise
    Does NOT terminate on reaching the goal; episodes end only due to:
      - falling into a hole (base env terminal), or
      - time limit (truncation wrapper in Gymnasium).
    Dynamics are identical across tasks; only rewards differ.
    """
    def __init__(self, env, goal_state: int):
        super().__init__(env)
        self.goal_state = goal_state

    def step(self, action):
        obs, r_env, terminated_env, truncated_env, info = self.env.step(action)

        # obs is a discrete state index for FrozenLake
        s = int(obs) if not isinstance(obs, (tuple, list)) else int(obs[0])

        # our custom reward: 1 if in goal_state, else 0
        reward = 1.0 if s == self.goal_state else 0.0

        # DO NOT terminate when on our goal_state; only respect env's own
        # termination (holes) and truncation (time limit).
        terminated = terminated_env
        truncated = truncated_env

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# -------- helper: get all non-hole states as possible goals --------
def get_possible_goal_states(custom_map):
    """
    Returns a list of state indices (0..15) that are not holes ('H').
    """
    possible_goals = []
    n_rows = len(custom_map)
    n_cols = len(custom_map[0])

    for r in range(n_rows):
        for c in range(n_cols):
            ch = custom_map[r][c]
            if ch != 'H':  # S or F are allowed as goals
                state_idx = r * n_cols + c
                possible_goals.append(state_idx)
    return possible_goals


# -------- make sure output folders exist --------
phi_folder = "phi"   # will store psi tables, one per goal (as requested)
w_folder = "w"

os.makedirs(phi_folder, exist_ok=True)
os.makedirs(w_folder, exist_ok=True)


# -------- training function for a single goal --------
def train_for_goal(goal_state: int):
    print(f"\n=== Training for goal_state = {goal_state} ===")

    # base FrozenLake env with custom map (no built-in G)
    base_env = gym.make(
        env_id,
        desc=custom_map,          # our fixed map, same for all tasks
        is_slippery=is_slippery,
        render_mode=None,         # no video for speed
    )

    # wrap with our task-specific reward (same dynamics, different reward)
    env = TaskRewardWrapper(base_env, goal_state=goal_state)

    # Gymnasium reset
    obs, info = env.reset()
    s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

    # ---- tables ----
    S = env.observation_space.n
    A = env.action_space.n
    d = S * A

    psi = torch.zeros((S, A, d), dtype=torch.float32)
    w = torch.zeros((d,), dtype=torch.float32)

    with tqdm(total=num_episodes, desc=f"Goal {goal_state} | Episodes", ncols=110) as epbar:
        for episode in range(1, num_episodes + 1):

            # epsilon schedule:
            decay_episodes = 1800
            if episode <= decay_episodes:
                frac = (episode - 1) / max(1, decay_episodes - 1)
                epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            else:
                epsilon = epsilon_end

            ep_return = 0.0
            w_errors = []
            psi_error_norms = []

            for step in range(max_steps_per_episode):

                # epsilon-greedy action (current state only)
                q_vals_curr = torch.matmul(psi[s], w)
                if random.random() < epsilon:
                    a = random.randrange(A)
                else:
                    a = int(torch.argmax(q_vals_curr).item())

                # environment step (uses our custom reward)
                step_result = env.step(a)
                if len(step_result) == 5:
                    s_dash_raw, r, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    s_dash_raw, r, done, info = step_result

                s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

                # greedy next action for SF target
                q_vals_next = torch.matmul(psi[s_dash], w)
                a_dash = int(torch.argmax(q_vals_next).item())

                # one-hot φ
                idx = int(s) * A + int(a)
                phi_vec = torch.zeros(d)
                phi_vec[idx] = 1.0

                # TD errors
                w_error = float(r - float(w[idx].item()))
                psi_pred = psi[s, a].clone()
                psi_target = phi_vec + gamma * psi[s_dash, a_dash]
                psi_error = psi_target - psi_pred
                psi_error_norm = float(torch.norm(psi_error).item())

                # store episode stats
                w_errors.append(w_error)
                psi_error_norms.append(psi_error_norm)

                # updates
                w[idx] += alpha_w * (r - w[idx])
                psi[s, a] += alpha_psi * psi_error

                ep_return += float(r)
                s = s_dash

                if done:
                    break

            # mean episode stats
            mean_w = sum(w_errors) / len(w_errors) if w_errors else 0.0
            mean_psi = sum(psi_error_norms) / len(psi_error_norms) if psi_error_norms else 0.0

            print(
                f"Goal {goal_state:2d} | Ep {episode:4d} | "
                f"w_err={mean_w:+.5f} | psi_err={mean_psi:.5f} | "
                f"return={ep_return:.1f} | eps={epsilon:.4f}"
            )

            # Gymnasium reset
            obs, info = env.reset()
            s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

            epbar.update(1)

    # ---- save artifacts for this goal ----
    phi_path = os.path.join(phi_folder, f"{goal_state}.pth")
    w_path = os.path.join(w_folder, f"{goal_state}.pth")

    # Note: we save the SF table psi under folder "phi" as requested
    torch.save(psi, phi_path)
    torch.save(w, w_path)

    print(f"Saved SF table (psi) for goal {goal_state} to {phi_path}")
    print(f"Saved w for goal {goal_state} to {w_path}")


# -------- main: train for all possible goal states --------
if __name__ == "__main__":
    possible_goals = get_possible_goal_states(custom_map)
    print("Possible goal states (non-hole tiles):", possible_goals)

    for g in possible_goals:
        train_for_goal(g)

    print("\nAll goals trained. SF tables saved in ./phi and w-vectors in ./w")
