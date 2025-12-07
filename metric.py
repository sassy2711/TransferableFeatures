# compare_q_tables.py
import os
import torch

# ---- paths (adjust folder names if needed) ----

# Per-goal "ground truth" psi* and its own w*
psi_star_dir = "psi"       # psi*/goal: psi/<goal>.pth
w_star_dir = "w"           # w*/goal:   w/<goal>.pth

# Shared psi for goal 9, without decorrelation, and its per-goal w's
psi_without_path = "psi_without_decorr(less_episodes).pth"
w_no_dir = "w's_for_psi9_star"   # per-goal w's for psi_without_decorr

# Shared psi for goal 9, with decorrelation, and its per-goal w's
psi_with_path = "psi_with_decorr(less_episodes).pth"
w_with_dir = "w's_for_psi_bar"   # per-goal w's for psi_with_decorr


def get_goal_ids(*dirs_and_labels):
    """
    From several (dir, label) pairs, collect goal IDs that exist as <id>.pth
    in ALL of the dirs. Returns sorted list of goal IDs and prints info.
    """
    id_sets = []
    for directory, label in dirs_and_labels:
        ids = set()
        if os.path.isdir(directory):
            for fname in os.listdir(directory):
                if fname.endswith(".pth"):
                    stem = os.path.splitext(fname)[0]
                    if stem.isdigit():
                        ids.add(int(stem))
        else:
            print(f"WARNING: directory '{directory}' for {label} does not exist.")
        print(f"{label} goals from '{directory}': {sorted(ids)}")
        id_sets.append(ids)

    # intersection across all sets
    common_ids = sorted(set.intersection(*id_sets)) if id_sets else []
    return common_ids


def main():
    # ---- load shared psi's ----
    psi_without = torch.load(psi_without_path).float()  # (S, A, d)
    psi_with = torch.load(psi_with_path).float()        # (S, A, d)

    # ---- find goals that exist in all four per-goal sets ----
    goals = get_goal_ids(
        (psi_star_dir, "psi*"),
        (w_star_dir, "w*"),
        (w_no_dir, "w_no (for psi_without)"),
        (w_with_dir, "w_with (for psi_with)"),
    )

    if not goals:
        print("\nNo common goals across psi*, w*, w_no, and w_with.")
        return

    print("\nCommon goals found:", goals)

    # sanity: determine S, A, d from first psi* and compare to shared psi's
    first_goal = goals[0]
    psi_star_sample = torch.load(os.path.join(psi_star_dir, f"{first_goal}.pth")).float()
    S, A, d = psi_star_sample.shape

    assert psi_without.shape == (S, A, d), \
        f"psi_without_decorr shape {psi_without.shape} != {(S, A, d)}"
    assert psi_with.shape == (S, A, d), \
        f"psi_with_decorr shape {psi_with.shape} != {(S, A, d)}"

    # ---- compute table rows ----
    rows = []
    for g in goals:
        psi_star_path = os.path.join(psi_star_dir, f"{g}.pth")
        w_star_path = os.path.join(w_star_dir, f"{g}.pth")
        w_no_path = os.path.join(w_no_dir, f"{g}.pth")
        w_with_path = os.path.join(w_with_dir, f"{g}.pth")

        print(f"\nProcessing goal {g}:")
        print("  psi*     :", psi_star_path)
        print("  w*       :", w_star_path)
        print("  w_no     :", w_no_path)
        print("  w_with   :", w_with_path)

        psi_star = torch.load(psi_star_path).float()  # (S, A, d)
        w_star = torch.load(w_star_path).float()      # (d,)
        w_no = torch.load(w_no_path).float()          # (d,)
        w_with_goal = torch.load(w_with_path).float() # (d,)

        # sanity checks
        assert psi_star.shape == (S, A, d), \
            f"psi* for goal {g} has shape {psi_star.shape}, expected {(S, A, d)}"
        for w_vec, name in [(w_star, "w*"), (w_no, "w_no"), (w_with_goal, "w_with")]:
            assert w_vec.ndim == 1 and w_vec.shape[0] == d, \
                f"{name} for goal {g} has shape {w_vec.shape}, expected ({d},)"

        # Q(s,a) = psi(s,a)^T w  -> shape (S, A)

        # "ground truth" for that goal
        Q_star = torch.tensordot(psi_star, w_star, dims=([2], [0]))        # (S, A)

        # no-decorr pair: same goal's w_no with psi_without
        Q_no = torch.tensordot(psi_without, w_no, dims=([2], [0]))         # (S, A)

        # decorr pair: same goal's w_with_goal with psi_with
        Q_with = torch.tensordot(psi_with, w_with_goal, dims=([2], [0]))   # (S, A)

        # Column 2: sum_{s,a} |Q* - Q_no|
        diff_no = torch.abs(Q_star - Q_no).sum().item()
        # Column 3: sum_{s,a} |Q* - Q_with|
        diff_with = torch.abs(Q_star - Q_with).sum().item()

        rows.append((g, diff_no, diff_with))

    # ---- print table ----
    print("\n================ Q-difference table ================")
    header = f"{'Goal':>4} | {'Sum |Q* - Q_no|':>20} | {'Sum |Q* - Q_with|':>20}"
    print(header)
    print("-" * len(header))
    for g, d2, d3 in rows:
        print(f"{g:4d} | {d2:20.6f} | {d3:20.6f}")


if __name__ == "__main__":
    main()
