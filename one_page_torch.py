import torch
import numpy as np

# Create a tensor
numpy_tensor = np.random.rand(10, 20)
t1 = torch.tensor(numpy_tensor)
t2 = torch.zeros(10, 20)
t3 = torch.ones(10, 20)

t4 = torch.rand(10, 20)
t4_with_different_generator = torch.rand(10, 20, generator=torch.Generator().manual_seed(42))
# arange = array range
t5 = torch.arange(0, 10, 1)
print(f"t5: {t5}, shape: {t5.shape}")
t5_but_3d = torch.arange(0, 10, 1).view(2, 5)
print(t5_but_3d.shape)

# Indexing and slicing
t6 = t5[1:5]
print(f"t6: {t6}")
# tensor[condition]
condition = t5 % 2 == 0
t7 = t5[condition]
print(f"t7: {t7}")


# Shape manipulation
t8 = t5.view(2, 5)
print(f"t8: {t8}")
t9 = torch.arange(0, 24, 1).reshape(2, 3, 4)
print(f"t9: {t9}")
t10 = t9.permute(1, 2, 0) # permute the dimensions => swap the dimensions .= 3,4,2
print(f"t10: {t10}", t10.shape)
## squeeze and unsqueeze
t11 = t5.unsqueeze(0)
print(f"t11: {t11}", t11.shape)
t12 = t11.squeeze(0)
print(f"t12: {t12}", t12.shape)
t13 = t10.flatten()
print(f"t13: {t13}", t13.shape)

# Element-wise operations => operations that are performed element by element
t14 = t5 + 2
print(f"t14: {t14}")
t15 = t5 * 2
print(f"t15: {t15}")
t16 = t5 / 2
print(f"t16: {t16}")
t17 = t15 * t16
print(f"t17: {t17}")
t18 = t17 ** (1/2)
print(f"t18: {t18}")
print(t18 == t5)

# Matrix operations
t19 = t5.dot(t5)
print(f"t19: {t19}")
t20 = t5 @ t5
print(f"t20: {t20}")
t21 = t8.t().matmul(t8)
print(f"t21: {t21}")

# Reduction operations
t22 = torch.sum(t8, dim=1)
print(f"t22: {t22}, shape: {t22.shape}")



if __name__ == "__main__":
    print("End of the program.")
