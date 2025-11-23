# # learn_w_from_fixed_psi_all_goals.py
# import gymnasium as gym
# import torch
# import os
# from tqdm import tqdm
# import random

# # -------- config --------
# env_id = "FrozenLake-v1"
# is_slippery = False
# num_episodes = 2000
# max_steps_per_episode = 100
# gamma = 0.99          # not directly used in w update here, but kept for clarity
# alpha_w = 0.05        # learning rate for w

# # epsilon schedule
# epsilon_start = 1.0
# epsilon_end = 0.01
# decay_episodes = 1800

# # ---- custom map with NO built-in G ----
# custom_map = [
#     "SFFF",
#     "FHFH",
#     "FFFH",
#     "HFFF",
# ]

# # --------- wrapper that gives reward for a given goal_state ---------
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


# # ---------- helper: all non-hole states as possible goals ----------
# def get_possible_goal_states(custom_map):
#     """
#     Returns a list of state indices (0..15) that are not holes ('H').
#     """
#     possible_goals = []
#     n_rows = len(custom_map)
#     n_cols = len(custom_map[0])
#     for r in range(n_rows):
#         for c in range(n_cols):
#             ch = custom_map[r][c]
#             if ch != 'H':  # allow 'S' and 'F'
#                 state_idx = r * n_cols + c
#                 possible_goals.append(state_idx)
#     return possible_goals


# # ---------- load fixed psi ----------
# psi_path = "psi_without_decorr.pth"
# psi = torch.load(psi_path).float()   # shape (S, A, d)

# S, A, d = psi.shape
# assert d == S * A, f"Expected d = S*A, got d={d}, S*A={S*A}"

# print(f"Loaded psi from {psi_path} with shape {psi.shape}")

# # ---------- folder for saving w's (rename this string if you want) ----------
# w_folder = "w's_for_psi9_star"
# os.makedirs(w_folder, exist_ok=True)


# def train_w_for_goal(goal_state: int):
#     print(f"\n===== Training w for goal_state = {goal_state} =====")

#     # build environment for this goal
#     base_env = gym.make(
#         env_id,
#         desc=custom_map,
#         is_slippery=is_slippery,
#     )
#     env = TaskRewardWrapper(base_env, goal_state=goal_state)

#     # sanity: psi dimensions vs env
#     S_env = env.observation_space.n
#     A_env = env.action_space.n
#     assert S_env == S, f"Env S={S_env} but psi S={S}"
#     assert A_env == A, f"Env A={A_env} but psi A={A}"

#     # initialize w freshly for this goal
#     w = torch.zeros(d, dtype=torch.float32)

#     obs, info = env.reset()
#     s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

#     with tqdm(total=num_episodes, desc=f"Goal {goal_state} | Learning w", ncols=110) as epbar:
#         for episode in range(1, num_episodes + 1):

#             # epsilon schedule
#             if episode <= decay_episodes:
#                 frac = (episode - 1) / max(1, decay_episodes - 1)
#                 epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
#             else:
#                 epsilon = epsilon_end

#             ep_return = 0.0
#             w_errors = []

#             for step in range(max_steps_per_episode):

#                 # epsilon-greedy action using fixed psi and current w
#                 q_vals_curr = torch.matmul(psi[s], w)   # shape (A,)
#                 a = random.randrange(A)

#                 step_result = env.step(a)
#                 if len(step_result) == 5:
#                     s_dash_raw, r, terminated, truncated, info = step_result
#                     done = terminated or truncated
#                 else:
#                     s_dash_raw, r, done, info = step_result

#                 s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

#                 # one-hot feature index for (s,a)
#                 idx = int(s) * A + int(a)

#                 # simple TD-style update for w[idx]:
#                 # we assume r ≈ w[idx] (since φ is one-hot for (s,a))
#                 w_error = float(r - float(w[idx].item()))
#                 w[idx] += alpha_w * w_error

#                 w_errors.append(w_error)
#                 ep_return += float(r)
#                 s = s_dash

#                 if done:
#                     break

#             mean_w_err = sum(w_errors) / len(w_errors) if w_errors else 0.0
#             print(
#                 f"Goal {goal_state:2d} | Episode {episode:4d} "
#                 f"| w_err={mean_w_err:+.5f} | return={ep_return:.1f} | eps={epsilon:.4f}"
#             )

#             obs, info = env.reset()
#             s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

#             epbar.update(1)

#     # save learned w for this goal
#     w_path = os.path.join(w_folder, f"{goal_state}.pth")
#     torch.save(w, w_path)
#     print(f"Saved w for goal_state={goal_state} to {w_path}")

#     env.close()


# def main():
#     possible_goals = get_possible_goal_states(custom_map)
#     print("Possible goal states (non-hole tiles):", possible_goals)

#     for g in possible_goals:
#         train_w_for_goal(g)

#     print("\nAll goals trained; w's saved in folder:", w_folder)


# if __name__ == "__main__":
#     main()


# learn_w_from_fixed_psi_all_goals.py
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

# ---- custom map with NO built-in G ----
custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFF",
]

# --------- wrapper that gives reward for a given goal_state ---------
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


# ---------- helper: all non-hole states as possible goals ----------
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
            if ch != 'H':  # allow 'S' and 'F'
                state_idx = r * n_cols + c
                possible_goals.append(state_idx)
    return possible_goals


# ---------- load fixed psi ----------
psi_path = "psi_with_decorr.pth"
psi = torch.load(psi_path).float()   # shape (S, A, d)

S, A, d = psi.shape
assert d == S * A, f"Expected d = S*A, got d={d}, S*A={S*A}"

print(f"Loaded psi from {psi_path} with shape {psi.shape}")

# ---------- folder for saving w's (rename this string if you want) ----------
w_folder = "w's_for_psi_bar"
os.makedirs(w_folder, exist_ok=True)


def train_w_for_goal(goal_state: int):
    print(f"\n===== Training w for goal_state = {goal_state} =====")

    # build environment for this goal
    base_env = gym.make(
        env_id,
        desc=custom_map,
        is_slippery=is_slippery,
    )
    env = TaskRewardWrapper(base_env, goal_state=goal_state)

    # sanity: psi dimensions vs env
    S_env = env.observation_space.n
    A_env = env.action_space.n
    assert S_env == S, f"Env S={S_env} but psi S={S}"
    assert A_env == A, f"Env A={A_env} but psi A={A}"

    # initialize w + visit counts for running-mean reward model
    w = torch.zeros(d, dtype=torch.float32)
    counts = torch.zeros(d, dtype=torch.float32)

    obs, info = env.reset()
    s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

    with tqdm(total=num_episodes, desc=f"Goal {goal_state} | Learning w", ncols=110) as epbar:
        for episode in range(1, num_episodes + 1):

            ep_return = 0.0
            w_errors = []

            for step in range(max_steps_per_episode):

                # PURE RANDOM ACTION (behavior policy)
                a = random.randrange(A)

                step_result = env.step(a)
                if len(step_result) == 5:
                    s_dash_raw, r, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    s_dash_raw, r, done, info = step_result

                s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

                # one-hot feature index for (s,a)
                idx = int(s) * A + int(a)

                # running mean update for reward model:
                # w[idx] ≈ E[r | s,a]
                counts[idx] += 1.0
                old_w = w[idx].item()
                w[idx] += (r - w[idx]) / counts[idx]

                # logging "error" as change from old value
                w_error = float(r - old_w)
                w_errors.append(w_error)

                ep_return += float(r)
                s = s_dash

                if done:
                    break

            mean_w_err = sum(w_errors) / len(w_errors) if w_errors else 0.0
            print(
                f"Goal {goal_state:2d} | Episode {episode:4d} "
                f"| w_err={mean_w_err:+.5f} | return={ep_return:.1f}"
            )

            obs, info = env.reset()
            s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

            epbar.update(1)

    # save learned w for this goal
    w_path = os.path.join(w_folder, f"{goal_state}.pth")
    torch.save(w, w_path)
    print(f"Saved w for goal_state={goal_state} to {w_path}")

    env.close()


def main():
    possible_goals = get_possible_goal_states(custom_map)
    print("Possible goal states (non-hole tiles):", possible_goals)

    for g in possible_goals:
        train_w_for_goal(g)

    print("\nAll goals trained; w's saved in folder:", w_folder)


if __name__ == "__main__":
    main()
