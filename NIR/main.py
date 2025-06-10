#
# import torch
#
# def get_mgrid(sidelen, vmin=-1, vmax=1):      #生成一个torch.tensor
#     x = torch.tensor([1])
#     y = torch.tensor([4,5])
#     z = torch.tensor([7,8,9])
#     m = torch.tensor([10,11,12,13])
#     print(torch.meshgrid(x,y,z,m,indexing='ij'))
#     mgrid = torch.stack(torch.meshgrid(x,y,z,indexing='ij'), dim=-1)
#     mgrid = mgrid.reshape(-1, len(sidelen))
#     return mgrid
#
#
# import torch
#
# x = torch.tensor([1.0, 2.0])       # 2 elements
# y = torch.tensor([3.0, 4.0, 5.0])  # 3 elements
# z = torch.tensor([6.0, 7.0])       # 2 elements
#
# X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')  # 从 PyTorch 1.10 推荐加 indexing 参数
#
# print("X.shape:", X.shape)
# print("Y.shape:", Y.shape)
# print("Z.shape:", Z.shape)
#
# if __name__ == '__main__':
#     #res = get_mgrid([3,3,3],-1,1)
#     #import torch
#
#     tensor = torch.randn(2, 3, 4, 4)  # 随机生成一个 4D 张量
#     print(tensor.shape)
#     print(tensor)

import torch

# 假设是时间步T1的输出
T1 = torch.tensor([[1, 2, 3],
        		[4, 5, 6],
        		[7, 8, 9]])
# 假设是时间步T2的输出
T2 = torch.tensor([[10, 20, 30],
        		[40, 50, 60],
        		[70, 80, 90]])


print(torch.cat([T1,T2],dim=1))

print(T1.shape)
print(T2.shape)

print(torch.stack((T1,T2),dim=0))
print(torch.stack((T1,T2),dim=1))
print(torch.stack((T1,T2),dim=2))


