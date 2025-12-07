import os
import torch

psi_folder = "psi"   # folder containing goal numbered .pth files
psi_star_path = "psi_without_decorr.pth"
psi_bar_path  = "psi_with_decorr.pth"

def l2_distance(a, b=None):
    """
    Computes L2 norm.
    If b is None → distance from origin.
    Else → distance between tensors a and b.
    """
    if b is None:
        v = a.reshape(-1).float()
    else:
        v = (a - b).reshape(-1).float()
    return torch.norm(v, p=2).item()

# Load psi_star and psi_bar
psi_star = torch.load(psi_star_path)
psi_bar  = torch.load(psi_bar_path)

results = []

# collect all numeric goal files
goal_files = []
for fname in os.listdir(psi_folder):
    if fname.endswith(".pth"):
        try:
            goal = int(fname.replace(".pth", ""))
            goal_files.append((goal, fname))
        except:
            pass

# sort by integer goal number
goal_files.sort(key=lambda x: x[0])

# compute distances
for goal, fname in goal_files:
    path = os.path.join(psi_folder, fname)
    psi = torch.load(path)

    dist_origin = l2_distance(psi)
    dist_star   = l2_distance(psi, psi_star)
    dist_bar    = l2_distance(psi, psi_bar)

    results.append((goal, dist_origin, dist_star, dist_bar))

# print table
print(f"{'Goal':<6} {'||psi||':<18} {'||psi-psi_star||':<20} {'||psi-psi_bar||':<20}")
print("-" * 70)

for goal, d0, dstar, dbar in results:
    print(f"{goal:<6} {d0:<18.6f} {dstar:<20.6f} {dbar:<20.6f}")
