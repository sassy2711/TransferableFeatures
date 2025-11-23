import torch

# paths to your saved psi tensors
path1 = "psi_wth.pth"
path2 = "psi_without.pth"

# load them
psi1 = torch.load(path1)
psi2 = torch.load(path2)

# ensure they are tensors
psi1 = psi1.float()
psi2 = psi2.float()

# check shape
assert psi1.shape == psi2.shape, f"Shape mismatch: {psi1.shape} vs {psi2.shape}"

# compute element-wise mean squared difference
mse = torch.mean((psi1 - psi2) ** 2)

print("Mean squared difference between ψ1 and ψ2:", mse.item())
