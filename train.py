# # # # import torch
# # # # import random

# # # # # dims and hyperparams
# # # # S = 8            # number of states
# # # # A = 3            # number of actions
# # # # d = S * A * S    # feature dim = number of (s,a,s') triples
# # # # gamma = 0.95
# # # # alpha_psi = 0.2
# # # # alpha_w = 0.1
# # # # device = "cpu"

# # # # # initialize tables (psi and w)
# # # # psi = torch.zeros((S, A, d), dtype=torch.float32, device=device)  # psi[s,a] is a length-d vector
# # # # w = torch.zeros((d,), dtype=torch.float32, device=device)         # reward weights for phi
# # # # num_steps = 300

# # # # for t in range(num_steps):
# # # #     #get current state from environment or have some starting state
# # # #     s = random.randrange(S)                  # current state

# # # #     #greedy action selection for getting next state and reward
# # # #     q_vals = torch.matmul(psi[s], w)       # shape (A,)
# # # #     a = int(torch.argmax(q_vals).item())

# # # #     # next state and reward should come after taking a_star in s
# # # #     s_dash = 1 
# # # #     r = 0 

# # # #     #check if s_dash is a terminal state, if yes then we will end here....
# # # #     q_vals_next = torch.matmul(psi[s_dash], w)       # shape (A,)
# # # #     #otherwise greedily choose a_dash as well for the psi update
# # # #     a_dash = int(torch.argmax(q_vals_next).item())

# # # #     # create phi vector (one-hot)
# # # #     phi = torch.zeros(d, dtype=torch.float32, device=device)
# # # #     idx = int(s) * (A * S) + int(a) * S + int(s_dash)   # integer in [0, d-1]
# # # #     phi[idx] = 1.0

# # # #     # w update
# # # #     w[idx] += alpha_w*(r - w[idx])

# # # #     #psi update
# # # #     psi[s, a] += alpha_psi*(phi + gamma*psi[s_dash, a_dash] - psi[s, a])



# # # import torch
# # # import random
# # # import gymnasium as gym

# # # # create deterministic FrozenLake (4x4) for easy tabular experiments
# # # env = gym.make("FrozenLake-v1", is_slippery=True)  # deterministic transitions
# # # # if using gymnasium, the same call should work in most setups

# # # # dims and hyperparams (kept your update rules exactly)
# # # S = env.observation_space.n    # number of states (16 for 4x4 FrozenLake)
# # # A = env.action_space.n         # number of actions (4)
# # # d = S * A * S                  # feature dim = number of (s,a,s') triples
# # # gamma = 0.99
# # # alpha_psi = 0.15
# # # alpha_w = 0.05
# # # device = "cpu"

# # # # initialize tables (psi and w)
# # # psi = torch.zeros((S, A, d), dtype=torch.float32, device=device)  # psi[s,a] is a length-d vector
# # # w = torch.zeros((d,), dtype=torch.float32, device=device)         # reward weights for phi

# # # num_steps = 10000

# # # # start in a reset environment
# # # obs = env.reset()
# # # # gym returns either obs or (obs, info) depending on version; handle both
# # # if isinstance(obs, tuple):
# # #     s = int(obs[0])
# # # else:
# # #     s = int(obs)

# # # for t in range(num_steps):
# # #     # greedy action selection using current psi and w (unchanged from your code)
# # #     q_vals = torch.matmul(psi[s], w)       # shape (A,)
# # #     a = int(torch.argmax(q_vals).item())

# # #     # take action a in the environment
# # #     step_result = env.step(a)
# # #     # gym versions differ; handle both older and newer APIs:
# # #     # older: (s_dash, r, done, info)
# # #     # newer gymnasium: (s_dash, r, terminated, truncated, info)
# # #     if len(step_result) == 4:
# # #         s_dash_raw, r, done, info = step_result
# # #     else:
# # #         s_dash_raw, r, terminated, truncated, info = step_result
# # #         done = terminated or truncated

# # #     # normalize s_dash to integer state
# # #     if isinstance(s_dash_raw, tuple) or isinstance(s_dash_raw, list):
# # #         s_dash = int(s_dash_raw[0])
# # #     else:
# # #         s_dash = int(s_dash_raw)

# # #     # q_vals_next and a_dash (keep exactly as in your pseudo-code)
# # #     q_vals_next = torch.matmul(psi[s_dash], w)       # shape (A,)
# # #     a_dash = int(torch.argmax(q_vals_next).item())

# # #     # create phi vector (one-hot) exactly as you wrote
# # #     phi = torch.zeros(d, dtype=torch.float32, device=device)
# # #     idx = int(s) * (A * S) + int(a) * S + int(s_dash)   # integer in [0, d-1]
# # #     phi[idx] = 1.0

# # #     # w update (unchanged)
# # #     w[idx] += alpha_w * (r - w[idx])

# # #     # psi update (unchanged)
# # #     psi[s, a] += alpha_psi * (phi + gamma * psi[s_dash, a_dash] - psi[s, a])

# # #     # move to next state; if terminal, reset environment to start a new episode
# # #     if done:
# # #         obs = env.reset()
# # #         if isinstance(obs, tuple):
# # #             s = int(obs[0])
# # #         else:
# # #             s = int(obs)
# # #     else:
# # #         s = s_dash

# # # # optionally: print small diagnostics
# # # print("Finished steps. psi shape:", psi.shape, "w length:", w.shape[0])
# # # # Show a few non-zero entries of w (sparse because of one-hot updates)
# # # nonzero = (w.abs() > 1e-6).nonzero(as_tuple=False)
# # # print("Number of nonzero w entries:", nonzero.shape[0])
# # # if nonzero.shape[0] > 0:
# # #     print("Sample nonzero indices and values:", [(int(i), float(w[int(i)])) for i in nonzero[:10].squeeze(1)])


# # # frozenlake_sf_with_tqdm_and_video.py
# # import gymnasium as gym
# # import torch
# # import random
# # import os
# # from tqdm import tqdm

# # # ----------------------------
# # # Config
# # # ----------------------------
# # device = "cpu"
# # env_id = "FrozenLake-v1"
# # is_slippery = True          # keep your setting
# # num_steps = 10000
# # gamma = 0.99
# # alpha_psi = 0.15
# # alpha_w = 0.05

# # # Video settings
# # video_folder = "./videos"
# # os.makedirs(video_folder, exist_ok=True)
# # record_interval = 10   # record every Nth episode (i.e., record episode when (episode % record_interval)==0)
# # # Implementation detail: we attach a RecordVideo wrapper when we want to record the *next* episode,
# # # then remove it after that episode finishes. This keeps recording logic local and explicit.

# # # ----------------------------
# # # Helper: create env (optionally wrapped for recording next episode)
# # # ----------------------------
# # def make_env(record_next_episode: bool = False):
# #     if record_next_episode:
# #         # render_mode must be "rgb_array" for RecordVideo to work
# #         base = gym.make(env_id, is_slippery=is_slippery, render_mode="rgb_array")
# #         # the wrapper records only the first episode after creation if we set episode_trigger for ep==0
# #         wrapped = gym.wrappers.RecordVideo(
# #             base,
# #             video_folder,
# #             episode_trigger=lambda ep: ep == 0,   # record the *first* episode seen by this wrapper
# #             name_prefix="frozenlake_episode"
# #         )
# #         return wrapped
# #     else:
# #         # normal env without rendering
# #         return gym.make(env_id, is_slippery=is_slippery)

# # # ----------------------------
# # # Init env and tables (kept your math & shapes exactly)
# # # ----------------------------
# # env = make_env(record_next_episode=False)

# # # get state/action counts from env
# # # for gymnasium reset() returns (obs, info)
# # obs, info = env.reset()
# # if isinstance(obs, (tuple, list)):
# #     s = int(obs[0])
# # else:
# #     s = int(obs)

# # S = env.observation_space.n
# # A = env.action_space.n
# # d = S * A * S

# # psi = torch.zeros((S, A, d), dtype=torch.float32, device=device)
# # w = torch.zeros((d,), dtype=torch.float32, device=device)

# # # ----------------------------
# # # Training loop with tqdm & per-episode logging
# # # ----------------------------
# # total_steps = 0
# # episode = 0

# # # metrics accumulators (optional)
# # episode_logs = []

# # pbar = tqdm(total=num_steps, desc="Training steps", ncols=100)

# # # keep track whether the current env is a recorder wrapper (so we can remove it after recording)
# # recorder_attached = False

# # # initial state s already computed above
# # while total_steps < num_steps:
# #     # greedy selection (unchanged)
# #     q_vals = torch.matmul(psi[s], w)       # (A,)
# #     a = int(torch.argmax(q_vals).item())

# #     # step
# #     step_result = env.step(a)
# #     # gymnasium: (obs, reward, terminated, truncated, info)
# #     if len(step_result) == 5:
# #         s_dash_raw, r, terminated, truncated, info = step_result
# #         done = terminated or truncated
# #     else:
# #         # fallback (older gym)
# #         s_dash_raw, r, done, info = step_result

# #     if isinstance(s_dash_raw, (tuple, list)):
# #         s_dash = int(s_dash_raw[0])
# #     else:
# #         s_dash = int(s_dash_raw)

# #     # compute next greedy action
# #     q_vals_next = torch.matmul(psi[s_dash], w)
# #     a_dash = int(torch.argmax(q_vals_next).item())

# #     # create phi one-hot index and phi vector (exactly your math)
# #     idx = int(s) * (A * S) + int(a) * S + int(s_dash)
# #     phi = torch.zeros(d, dtype=torch.float32, device=device)
# #     phi[idx] = 1.0

# #     # --- compute TD-errors BEFORE updates (for logging) ---
# #     w_pred_before = float(w[idx].item())
# #     w_tgt = float(r)
# #     w_error = w_tgt - w_pred_before

# #     psi_pred_before = psi[s, a].clone()
# #     psi_target = phi + gamma * psi[s_dash, a_dash]
# #     psi_error_vec = psi_target - psi_pred_before
# #     psi_error_norm = float(torch.norm(psi_error_vec).item())

# #     # --- updates (unchanged) ---
# #     w[idx] += alpha_w * (w_tgt - w[idx])
# #     psi[s, a] += alpha_psi * psi_error_vec

# #     total_steps += 1
# #     pbar.update(1)

# #     # maintain some running postfix info
# #     pbar.set_postfix(step=total_steps)

# #     # if terminal, log episode info and reset. Also handle video wrapper removal/attachment.
# #     if done:
# #         episode += 1
# #         ep_return = float(r)  # FrozenLake gives only final reward usually 0 or 1; you could accumulate if you want
# #         ep_len = None  # we don't keep per-episode step count here but you can add if desired

# #         # Log summary for this terminal episode
# #         print(f"\nEpisode {episode} finished | w_error={w_error:.4f} | psi_error_norm={psi_error_norm:.4f} | final_reward={ep_return}")

# #         episode_logs.append({
# #             "episode": episode,
# #             "w_error": w_error,
# #             "psi_error_norm": psi_error_norm,
# #             "final_reward": ep_return
# #         })

# #         # If we had attached a recorder for this episode, the video is now saved in video_folder.
# #         # Remove the recorder and replace env with a fresh non-recording env for next episodes.
# #         if recorder_attached:
# #             try:
# #                 env.close()
# #             except Exception:
# #                 pass
# #             env = make_env(record_next_episode=False)
# #             recorder_attached = False

# #         # Decide whether to attach recorder for next episode
# #         if (episode % record_interval) == 0:
# #             # attach recorder so that the very next episode is recorded
# #             try:
# #                 env.close()
# #             except Exception:
# #                 pass
# #             env = make_env(record_next_episode=True)
# #             recorder_attached = True

# #         # reset env (gymnasium reset returns obs, info)
# #         obs, info = env.reset()
# #         if isinstance(obs, (tuple, list)):
# #             s = int(obs[0])
# #         else:
# #             s = int(obs)
# #     else:
# #         s = s_dash

# # pbar.close()

# # # ----------------------------
# # # Save artifacts and print summary
# # # ----------------------------
# # torch.save(psi.cpu(), "psi_table_frozenlake.pth")
# # torch.save(w.detach().cpu(), "w_frozenlake.pth")
# # print(f"\nSaved psi_table_frozenlake.pth and w_frozenlake.pth. Total episodes: {episode}.")
# # print(f"Recorded videos (every {record_interval} episodes) saved to: {os.path.abspath(video_folder)}")


# # frozenlake_sf_fixed_video.py
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

# # epsilon schedule
# epsilon_start = 1.0
# epsilon_end = 0.01

# # -------- VIDEO FOLDER (overwrite allowed) --------
# video_folder = "./videos"
# if not os.path.exists(video_folder):
#     os.makedirs(video_folder)

# record_interval = 100   # record every 100 episodes

# print("📹 Videos will be saved to:", os.path.abspath(video_folder))
# print("   Recording every", record_interval, "episodes")

# # -------- create SINGLE persistent env with video wrapper --------
# def should_record(ep_id):
#     return (ep_id % record_interval) == 0

# # internal episode counter for RecordVideo starts at 0 for wrapper
# env = gym.make(env_id, is_slippery=is_slippery, render_mode="rgb_array")
# env = gym.wrappers.RecordVideo(
#     env,
#     video_folder,
#     episode_trigger=should_record,
#     name_prefix="frozenlake"
# )

# # Gymnasium reset
# obs, info = env.reset()
# s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

# # -------- tables --------
# S = env.observation_space.n
# A = env.action_space.n
# d = S * A * S

# psi = torch.zeros((S, A, d), dtype=torch.float32)
# w = torch.zeros((d,), dtype=torch.float32)

# # -------- training --------
# with tqdm(total=num_episodes, desc="Episodes", ncols=110) as epbar:
#     for episode in range(1, num_episodes + 1):

#         # epsilon schedule:
#         # - linearly decay from 1.0 to 0.01 over the first 14000 episodes
#         # - then keep it fixed at 0.01 for the last 1000 episodes
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

#             # epsilon-greedy action (current state only)
#             q_vals_curr = torch.matmul(psi[s], w)
#             if random.random() < epsilon:
#                 a = random.randrange(A)
#             else:
#                 a = int(torch.argmax(q_vals_curr).item())

#             # environment step
#             step_result = env.step(a)
#             if len(step_result) == 5:
#                 s_dash_raw, r, terminated, truncated, info = step_result
#                 done = terminated or truncated
#             else:
#                 s_dash_raw, r, done, info = step_result

#             s_dash = int(s_dash_raw) if not isinstance(s_dash_raw, (tuple, list)) else int(s_dash_raw[0])

#             # greedy next action for SF target
#             q_vals_next = torch.matmul(psi[s_dash], w)
#             a_dash = int(torch.argmax(q_vals_next).item())

#             # one-hot φ
#             idx = int(s) * (A * S) + int(a) * S + int(s_dash)
#             phi = torch.zeros(d)
#             phi[idx] = 1.0

#             # TD errors
#             w_error = float(r - float(w[idx].item()))
#             psi_pred = psi[s, a].clone()
#             psi_target = phi + gamma * psi[s_dash, a_dash]
#             psi_error = psi_target - psi_pred
#             psi_error_norm = float(torch.norm(psi_error).item())

#             # store episode stats
#             w_errors.append(w_error)
#             psi_error_norms.append(psi_error_norm)

#             # updates
#             w[idx] += alpha_w * (r - w[idx])
#             psi[s, a] += alpha_psi * psi_error

#             ep_return += float(r)
#             s = s_dash

#             if done:
#                 break

#         # mean episode stats
#         mean_w = sum(w_errors) / len(w_errors)
#         mean_psi = sum(psi_error_norms) / len(psi_error_norms)

#         print(f"Episode {episode:4d} | w_err={mean_w:+.5f} | psi_err={mean_psi:.5f} | return={ep_return:.1f} | eps={epsilon:.4f}")

#         # Gymnasium reset
#         obs, info = env.reset()
#         s = int(obs[0]) if isinstance(obs, (tuple, list)) else int(obs)

#         epbar.update(1)

# # -------- save artifacts --------
# torch.save(psi, "psi_table_frozenlake.pth")
# torch.save(w, "w_frozenlake.pth")
# print("\nSaved psi_table_frozenlake.pth and w_frozenlake.pth")
# print("Videos saved in:", os.path.abspath(video_folder))


# frozenlake_sf_fixed_video.py
import gymnasium as gym
import torch
import os
from tqdm import tqdm
import random

# -------- config --------
env_id = "FrozenLake-v1"
is_slippery = False
num_episodes = 15000
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

record_interval = 100   # record every 100 episodes

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
d = S * A * S

psi = torch.zeros((S, A, d), dtype=torch.float32)
w = torch.zeros((d,), dtype=torch.float32)

# -------- training --------
with tqdm(total=num_episodes, desc="Episodes", ncols=110) as epbar:
    for episode in range(1, num_episodes + 1):

        # epsilon schedule:
        # - linearly decay from 1.0 to 0.01 over the first 14000 episodes
        # - then keep it fixed at 0.01 for the last 1000 episodes
        decay_episodes = 14000
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
            idx = int(s) * (A * S) + int(a) * S + int(s_dash)
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
