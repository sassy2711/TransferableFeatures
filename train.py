# frozenlake_sf_fixed_video.py
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
# Default 4x4 map is:
#   "SFFF",
#   "FHFH",
#   "FFFH",
#   "HFFG"
# We turn the G into F so there is no built-in goal:
custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFF",
]

# ---- task-specific goal (non-corner, non-hole state) ----
# State indexing:
#   0  1  2  3
#   4  5  6  7
#   8  9 10 11
#  12 13 14 15
# Holes in our custom map: 5, 7, 11, 12
# We'll pick 9 as our "goal" for this task (safe interior tile).
TASK_GOAL_STATE = 9


class TaskRewardWrapper(gym.Wrapper):
    """
    Wraps FrozenLake to define our own reward:
      - reward = 1 when the agent is in TASK_GOAL_STATE
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


# -------- VIDEO FOLDER (overwrite allowed) --------
video_folder = "./videos"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

record_interval = 1000   # record every 100 episodes

print("📹 Videos will be saved to:", os.path.abspath(video_folder))
print("   Recording every", record_interval, "episodes")

# -------- create SINGLE persistent env with wrappers --------
def should_record(ep_id):
    return (ep_id % record_interval) == 0

# base FrozenLake env with custom map (no built-in G)
base_env = gym.make(
    env_id,
    desc=custom_map,          # our fixed map, same for all tasks
    is_slippery=is_slippery,
    render_mode="rgb_array",
)

# wrap with our task-specific reward (same dynamics, different reward)
task_env = TaskRewardWrapper(base_env, goal_state=TASK_GOAL_STATE)

# then wrap with video recorder
env = gym.wrappers.RecordVideo(
    task_env,
    video_folder,
    episode_trigger=should_record,
    name_prefix="frozenlake"
)

# Gymnasium reset
obs, info = env.reset()
s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

# -------- tables --------
S = env.observation_space.n
A = env.action_space.n
d = S * A 

psi = torch.zeros((S, A, d), dtype=torch.float32)
w = torch.zeros((d,), dtype=torch.float32)

# -------- training --------
with tqdm(total=num_episodes, desc="Episodes", ncols=110) as epbar:
    for episode in range(1, num_episodes + 1):

        # epsilon schedule:
        # - linearly decay from 1.0 to 0.01 over the first 14000 episodes
        # - then keep it fixed at 0.01 for the last 1000 episodes
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
            idx = int(s) * A  + int(a) 
            phi = torch.zeros(d)
            phi[idx] = 1.0

            # TD errors
            w_error = float(r - float(w[idx].item()))
            psi_pred = psi[s, a].clone()
            psi_target = phi + gamma * psi[s_dash, a_dash]
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

        print(f"Episode {episode:4d} | w_err={mean_w:+.5f} | psi_err={mean_psi:.5f} | return={ep_return:.1f} | eps={epsilon:.4f}")

        # Gymnasium reset
        obs, info = env.reset()
        s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

        epbar.update(1)

# -------- save artifacts --------
torch.save(psi, "psi_table_frozenlake.pth")
torch.save(w, "w_frozenlake.pth")
print("\nSaved psi_table_frozenlake.pth and w_frozenlake.pth")
print("Videos saved in:", os.path.abspath(video_folder))
