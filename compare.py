import torch

path1 = "psi_without_decorr(less_episodes).pth"
path2 = "psi_without_decorr.pth"

# path1 = "w_with_decorr_goal6.pth"
# path2 = "w_without_decorr_goal6.pth"

psi1 = torch.load(path1).float()
psi2 = torch.load(path2).float()

assert psi1.shape == psi2.shape, f"Shape mismatch: {psi1.shape} vs {psi2.shape}"

# element-wise absolute difference and sum
abs_diff_sum = torch.sum(torch.abs(psi1 - psi2))

print("Sum of absolute differences:", abs_diff_sum.item())

max_diff = (psi1 - psi2).abs().max().item()
min_diff = (psi1 - psi2).abs().min().item()

print("Max absolute difference:", max_diff)
print("Min absolute difference:", min_diff)

num_nonzero = torch.count_nonzero(psi1 - psi2).item()
print("Number of elements that differ:", num_nonzero)
