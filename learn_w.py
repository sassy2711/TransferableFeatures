# learn_w_from_fixed_psi.py
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
gamma = 0.99          # not directly used in w update here, but kept for clarity
alpha_w = 0.05        # learning rate for w

# epsilon schedule
epsilon_start = 1.0
epsilon_end = 0.01
decay_episodes = 1800

# ---- custom map with NO built-in G ----
custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFF",
]

# pick the goal state for THIS TASK
TASK_GOAL_STATE = 6   # change this for another task


class TaskRewardWrapper(gym.Wrapper):
    def __init__(self, env, goal_state: int):
        super().__init__(env)
        self.goal_state = goal_state

    def step(self, action):
        obs, r_env, terminated_env, truncated_env, info = self.env.step(action)
        s = int(obs) if not isinstance(obs, (tuple, list)) else int(obs[0])
        reward = 1.0 if s == self.goal_state else 0.0
        return obs, reward, terminated_env, truncated_env, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ---------- load fixed psi ----------
psi_path = "psi_with_decorr.pth"
psi = torch.load(psi_path).float()   # shape (S, A, d)

S, A, d = psi.shape
assert d == S * A, f"Expected d = S*A, got d={d}, S*A={S*A}"

print(f"Loaded psi from {psi_path} with shape {psi.shape}")

# ---------- initialize w ----------
w = torch.zeros(d, dtype=torch.float32)

# ---------- build environment ----------
base_env = gym.make(
    env_id,
    desc=custom_map,
    is_slippery=is_slippery,
)

env = TaskRewardWrapper(base_env, goal_state=TASK_GOAL_STATE)

obs, info = env.reset()
s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

# ========== TRAINING LOOP (only w is learned) ==========
with tqdm(total=num_episodes, desc="Learning w (fixed psi)", ncols=110) as epbar:
    for episode in range(1, num_episodes + 1):

        # epsilon schedule
        if episode <= decay_episodes:
            frac = (episode - 1) / max(1, decay_episodes - 1)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        else:
            epsilon = epsilon_end

        ep_return = 0.0
        w_errors = []

        for step in range(max_steps_per_episode):

            # epsilon-greedy action using fixed psi and current w
            q_vals_curr = torch.matmul(psi[s], w)   # shape (A,)
            if random.random() < epsilon:
                a = random.randrange(A)
            else:
                a = int(torch.argmax(q_vals_curr))

            step_result = env.step(a)
            if len(step_result) == 5:
                s_dash_raw, r, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                s_dash_raw, r, done, info = step_result

            s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

            # one-hot feature index for (s,a)
            idx = int(s) * A + int(a)

            # simple TD-style update for w[idx]:
            # we assume r ≈ w[idx] (since φ is one-hot for (s,a))
            w_error = float(r - float(w[idx].item()))
            w[idx] += alpha_w * w_error

            w_errors.append(w_error)
            ep_return += float(r)
            s = s_dash

            if done:
                break

        mean_w_err = sum(w_errors) / len(w_errors) if w_errors else 0.0
        print(f"Episode {episode:4d} | w_err={mean_w_err:+.5f} | return={ep_return:.1f} | eps={epsilon:.4f}")

        obs, info = env.reset()
        s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

        epbar.update(1)

# save learned w for this task
w_path = "w_with_decorr_goal6.pth"
torch.save(w, w_path)
print(f"\nSaved w to {w_path}")
