# # frozenlake_sf_fixed_video.py   (with decorrelation regularization)
# import gymnasium as gym
# import torch
# import os
# from tqdm import tqdm
# import random

# # -------- config --------
# env_id = "FrozenLake-v1"
# is_slippery = False
# num_episodes = 15000
# max_steps_per_episode = 100
# gamma = 0.99
# alpha_psi = 0.15
# alpha_w = 0.05

# # decorrelation strength  (x = η/α)
# x = 20   # <<--- NEW HYPERPARAMETER

# # epsilon schedule
# epsilon_start = 1.0
# epsilon_end = 0.01

# # ---- custom map with NO built-in G ----
# custom_map = [
#     "SFFF",
#     "FHFH",
#     "FFFH",
#     "HFFF",
# ]

# TASK_GOAL_STATE = 9


# class TaskRewardWrapper(gym.Wrapper):
#     def __init__(self, env, goal_state: int):
#         super().__init__(env)
#         self.goal_state = goal_state

#     def step(self, action):
#         obs, r_env, terminated_env, truncated_env, info = self.env.step(action)
#         s = int(obs) if not isinstance(obs, (tuple, list)) else int(obs[0])
#         reward = 1.0 if s == self.goal_state else 0.0
#         return obs, reward, terminated_env, truncated_env, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)


# video_folder = "./videos"
# if not os.path.exists(video_folder):
#     os.makedirs(video_folder)

# record_interval = 100   
# def should_record(ep_id): return (ep_id % record_interval) == 0

# base_env = gym.make(
#     env_id,
#     desc=custom_map,
#     is_slippery=is_slippery,
#     render_mode="rgb_array",
# )

# task_env = TaskRewardWrapper(base_env, goal_state=TASK_GOAL_STATE)

# env = gym.wrappers.RecordVideo(
#     task_env, video_folder,
#     episode_trigger=should_record,
#     name_prefix="frozenlake"
# )

# obs, info = env.reset()
# s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

# # -------- tables --------
# S = env.observation_space.n
# A = env.action_space.n
# d = S * A 

# psi = torch.zeros((S, A, d), dtype=torch.float32)
# w = torch.zeros((d,), dtype=torch.float32)

# N = S * A      # number of ψ-vectors for covariance


# # ========== TRAINING LOOP ==========
# with tqdm(total=num_episodes, desc="Episodes", ncols=110) as epbar:
#     for episode in range(1, num_episodes + 1):

#         # epsilon schedule
#         decay_episodes = 14000
#         if episode <= decay_episodes:
#             frac = (episode - 1) / max(1, decay_episodes - 1)
#             epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
#         else:
#             epsilon = epsilon_end

#         ep_return = 0.0
#         w_errors = []
#         psi_error_norms = []

#         for step in range(max_steps_per_episode):

#             # epsilon-greedy action
#             q_vals_curr = torch.matmul(psi[s], w)
#             a = random.randrange(A) if random.random() < epsilon else int(torch.argmax(q_vals_curr))

#             step_result = env.step(a)
#             if len(step_result) == 5:
#                 s_dash_raw, r, terminated, truncated, info = step_result
#                 done = terminated or truncated
#             else:
#                 s_dash_raw, r, done, info = step_result

#             s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

#             # greedy next action
#             q_vals_next = torch.matmul(psi[s_dash], w)
#             a_dash = int(torch.argmax(q_vals_next))

#             # one-hot φ
#             idx = int(s) * A  + int(a) 
#             phi = torch.zeros(d)
#             phi[idx] = 1.0

#             # TD errors
#             w_error = float(r - float(w[idx].item()))
#             psi_pred = psi[s, a].clone()
#             psi_target = phi + gamma * psi[s_dash, a_dash]
#             psi_error = psi_target - psi_pred
#             psi_error_norm = float(torch.norm(psi_error).item())

#             w_errors.append(w_error)
#             psi_error_norms.append(psi_error_norm)


#             # ============================================================
#             # === DECORRELATION GRADIENT (new term from the PDF) =========
#             # ============================================================

#             # reshape successor features to matrix Ψ of shape (N, d)
#             Psi = psi.reshape(N, d)

#             # mean Ψ̄
#             Psi_mean = Psi.mean(dim=0, keepdim=True)

#             # centered matrix
#             C = Psi - Psi_mean

#             # covariance
#             Cov = (C.T @ C) / N

#             # gradient dL/dΨ = 4/N * C @ (Cov - I)
#             I = torch.eye(d)
#             GradPsi = (4.0 / N) * (C @ (Cov - I))

#             # extract gradient for this (s,a)
#             row_index = s * A + a
#             grad_row = GradPsi[row_index]

#             if episode % 200 == 0 and step == 0:  # or any condition so it doesn't spam
#                 td_norm = torch.norm(psi_error).item()
#                 reg_norm = torch.norm(grad_row).item()
#                 print(f"[ep={episode}] ||TD||={td_norm:.5e}, ||grad_row||={reg_norm:.5e}, x*||grad_row||={(x*reg_norm):.5e}")


#             # ============================================================
#             # ========== FINAL UPDATE: Bellman + decorrelation ===========
#             # ============================================================

#             psi[s, a] += alpha_psi * (psi_error - x * grad_row)

#             # plain w update (unchanged)
#             w[idx] += alpha_w * (r - w[idx])

#             ep_return += float(r)
#             s = s_dash

#             if done:
#                 break

#         mean_w = sum(w_errors)/len(w_errors) if w_errors else 0.0
#         mean_psi = sum(psi_error_norms)/len(psi_error_norms) if psi_error_norms else 0.0

#         print(f"Episode {episode:4d} | w_err={mean_w:+.5f} | psi_err={mean_psi:.5f} | return={ep_return:.1f} | eps={epsilon:.4f}")

#         obs, info = env.reset()
#         s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

#         epbar.update(1)

# # save
# torch.save(psi, "psi_with_decorr.pth")
# torch.save(w, "w_frozenlake.pth")
# print("\nSaved psi_table_frozenlake.pth and w_frozenlake.pth")
# print("Videos saved in:", os.path.abspath(video_folder))


# frozenlake_sf_fixed_video.py   (with visitation counts)
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

# decorrelation strength  (x = η/α)
x = 20   # <<--- NEW HYPERPARAMETER

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

TASK_GOAL_STATE = 9


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


video_folder = "./videos"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

record_interval = 100
def should_record(ep_id): return (ep_id % record_interval) == 0

base_env = gym.make(
    env_id,
    desc=custom_map,
    is_slippery=is_slippery,
    render_mode="rgb_array",
)

task_env = TaskRewardWrapper(base_env, goal_state=TASK_GOAL_STATE)

env = gym.wrappers.RecordVideo(
    task_env, video_folder,
    episode_trigger=should_record,
    name_prefix="frozenlake"
)

obs, info = env.reset()
s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

# -------- tables --------
S = env.observation_space.n
A = env.action_space.n
d = S * A

psi = torch.zeros((S, A, d), dtype=torch.float32)
w = torch.zeros((d,), dtype=torch.float32)

N = S * A      # number of ψ-vectors for covariance

# -------- visitation counts --------
visit_counts = torch.zeros(S, dtype=torch.long)

# count the initial state at the very beginning
visit_counts[s] += 1

# ========== TRAINING LOOP ==========
with tqdm(total=num_episodes, desc="Episodes", ncols=110) as epbar:
    for episode in range(1, num_episodes + 1):

        # epsilon schedule
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

            # epsilon-greedy action
            q_vals_curr = torch.matmul(psi[s], w)
            a = random.randrange(A) if random.random() < epsilon else int(torch.argmax(q_vals_curr))

            step_result = env.step(a)
            if len(step_result) == 5:
                s_dash_raw, r, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                s_dash_raw, r, done, info = step_result

            s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

            # count visitation of the next state
            visit_counts[s_dash] += 1

            # greedy next action
            q_vals_next = torch.matmul(psi[s_dash], w)
            a_dash = int(torch.argmax(q_vals_next))

            # one-hot φ
            idx = int(s) * A + int(a)
            phi = torch.zeros(d)
            phi[idx] = 1.0

            # TD errors
            w_error = float(r - float(w[idx].item()))
            psi_pred = psi[s, a].clone()
            psi_target = phi + gamma * psi[s_dash, a_dash]
            psi_error = psi_target - psi_pred
            psi_error_norm = float(torch.norm(psi_error).item())

            w_errors.append(w_error)
            psi_error_norms.append(psi_error_norm)

            # ============================================================
            # === DECORRELATION GRADIENT (new term from the PDF) =========
            # ============================================================

            # reshape successor features to matrix Ψ of shape (N, d)
            Psi = psi.reshape(N, d)

            # mean Ψ̄
            Psi_mean = Psi.mean(dim=0, keepdim=True)

            # centered matrix
            C = Psi - Psi_mean

            # covariance
            Cov = (C.T @ C) / N

            # gradient dL/dΨ = 4/N * C @ (Cov - I)
            I = torch.eye(d)
            GradPsi = (4.0 / N) * (C @ (Cov - I))

            # extract gradient for this (s,a)
            row_index = s * A + a
            grad_row = GradPsi[row_index]

            if episode % 200 == 0 and step == 0:  # or any condition so it doesn't spam
                td_norm = torch.norm(psi_error).item()
                reg_norm = torch.norm(grad_row).item()
                print(f"[ep={episode}] ||TD||={td_norm:.5e}, ||grad_row||={reg_norm:.5e}, x*||grad_row||={(x*reg_norm):.5e}")

            # ============================================================
            # ========== FINAL UPDATE: Bellman (no decorrelation) ========
            # ============================================================

            psi[s, a] += alpha_psi * psi_error

            # plain w update (unchanged)
            w[idx] += alpha_w * (r - w[idx])

            ep_return += float(r)
            s = s_dash

            if done:
                break

        mean_w = sum(w_errors)/len(w_errors) if w_errors else 0.0
        mean_psi = sum(psi_error_norms)/len(psi_error_norms) if psi_error_norms else 0.0

        print(f"Episode {episode:4d} | w_err={mean_w:+.5f} | psi_err={mean_psi:.5f} | return={ep_return:.1f} | eps={epsilon:.4f}")

        # reset env + count the initial state of the next episode
        obs, info = env.reset()
        s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)
        visit_counts[s] += 1

        epbar.update(1)

# save
torch.save(psi, "psi_without_decorr(less_episodes).pth")
torch.save(w, "w_frozenlake.pth")
print("\nSaved psi_without_decorr.pth and w_frozenlake.pth")
print("Videos saved in:", os.path.abspath(video_folder))

# -------- print visitation stats --------
print("\nState visitation counts (over all training episodes):")
for state in range(S):
    print(f"State {state:2d}: {visit_counts[state].item()}")

print("\nVisitation counts as 4x4 grid (matching map layout):")
n_rows, n_cols = 4, 4  # since your map is 4x4
for r in range(n_rows):
    row_counts = [f"{visit_counts[r * n_cols + c].item():6d}" for c in range(n_cols)]
    print(" ".join(row_counts))
