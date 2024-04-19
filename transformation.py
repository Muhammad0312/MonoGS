import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

T = torch.eye(4) # gt_initial
rotation = R.from_quat([0.8851, 0.2362, -0.0898, -0.3909])
T[:3, :3] = torch.tensor(rotation.as_matrix())
T[:3, 3] = torch.tensor([1.3112, 0.8507, 1.5186])

T0 = torch.eye(4) # gt_final
rotation = R.from_quat([0.8978, 0.2746, -0.0950, -0.3310])
T0[:3, :3] = torch.tensor(rotation.as_matrix())
T0[:3, 3] = torch.tensor([0.9453, -0.0038, 1.2888])


#estimated_final
T1 = torch.eye(4)
rotation = R.from_quat([-0.076708093, -0.043557819, -0.007320210, 0.996074855])
T1[:3, :3] = torch.tensor(rotation.as_matrix())
T1[:3, 3] = torch.tensor([-0.775839567, 0.551486015, -0.415254742])

final = torch.matmul(T, T1)
print(final)
print(T0)
