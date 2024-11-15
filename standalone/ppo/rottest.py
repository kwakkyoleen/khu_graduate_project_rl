import torch

# 예제 (n, 3) 크기의 벡터 텐서 생성
n = 5  # 벡터의 수
vecs = torch.randn(n, 3)  # (n, 3) 모양의 텐서

# x, y 좌표 추출
x = vecs[:, 0]
y = vecs[:, 1]

# y가 0이 되도록 회전할 각도 계산
theta_z = torch.atan2(y, x)  # (n,) 모양

# 각 벡터에 대한 회전 행렬 생성
cos_theta = torch.cos(theta_z)
sin_theta = torch.sin(theta_z)

# 회전 행렬을 각 벡터에 적용하기 위해 배치로 생성: (n, 3, 3)
Rz = torch.zeros((n, 3, 3))
Rz[:, 0, 0] = cos_theta
Rz[:, 0, 1] = sin_theta
Rz[:, 1, 0] = -sin_theta
Rz[:, 1, 1] = cos_theta
Rz[:, 2, 2] = 1

# 각 회전 행렬을 (n, 3) 텐서의 각 벡터에 적용
rotated_vecs = torch.bmm(Rz, vecs.unsqueeze(2)).squeeze(2)  # (n, 3)

# 결과 출력
print("회전 전 벡터:\n", vecs)
print("회전 후 벡터 (y 성분이 0에 가까워야 함):\n", rotated_vecs)