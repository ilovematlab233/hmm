#——————————————————————————————————HMM问题————————————————————————————————

# 导入库
import numpy as np


# ————————————————————————————————初始化数据——————————————————————————————

# 状态转移矩阵A
A = [[0.5, 0.1, 0.4],
     [0.3, 0.5, 0.2],
     [0.2, 0.2, 0.6]]

# 观测概率矩阵B
B = [[0.5, 0.5],
     [0.4, 0.6],
     [0.7, 0.3]]

# 初始状态取各个值的概率
Pi = [
    [0.2, 0.3, 0.5]]

# 初始化观测序列：0为红球，1为白球
O = [0, 1, 0, 0, 1, 0, 1, 1]

# 状态序列的大小
# N = len(Q)
N = 3

# 观测序列的大小
T = len(O)


# —————————————————————————————————前向算法————————————————————————————————

# 初始化前向概率alpha值
alpha = np.zeros((N, T))

# 遍历每一个时刻，计算alpha
for t in range(T):  # t = 0~7

    # 得到对应时间的结果
    Of = O[t]

    # 遍历状态序列
    for i in range(N):
        if t == 0:  # 初始化alpha初值
            alpha[i][t] = Pi[t][i] * B[i][Of]
        else:   # 根据课本公式，计算出alpha的转移
            alpha[i][t] = np.dot([alpha[t - 1] for alpha in alpha],[a[i] for a in A]) * B[i][Of]

# 计算前向结果
forward_P = np.sum([a[T - 1] for a in alpha])
print('1. 前向计算结果为：',forward_P)


# ——————————————————————————————————后向算法——————————————————————————————————

# 初始化后向概率beta值
betas = np.ones((N, T))

# 逆向遍历观测序列，计算beta
for t in range(T - 2, -1, -1):

    # 得到对应时间的结果
    Of = O[t+1]

    # 遍历状态序列
    for i in range(N):
        # 通过课本公式迭代betas值
        betas[i][t] = np.dot(np.multiply(A[i], [b[Of] for b in B]),
            [beta[t + 1] for beta in betas])

# 取出第一个值
Of = O[0]

# 计算后向结果
P = np.dot(np.multiply(Pi, [b[Of] for b in B]),[beta[0] for beta in betas])

backward_P = P
print("2. 后向计算结果为：",backward_P[0])


# ——————————————————————————————使用前向后向概率计算————————————————————————————————

result = (alpha[2][3] * betas[2][3]) / backward_P[0]
print("3. 前向后向概率计算为：", result)


# ——————————————————————————————维特比算法求最优路径————————————————————————————————

# 初始化daltas
deltas = np.zeros((N, T))

# 初始化psis
psis = np.zeros((N, T))

# 初始化最优路径为0
I = np.zeros((1, T))

# 遍历观测序列
for t in range(T):

    # 从2开始递推
    realT = t + 1

    # 得到序列对应的索引
    Of = O[t]

    # 开始进行遍历
    for i in range(N):

        if t == 0:  # 初始化deltas值
            deltas[i][t] = Pi[0][i] * B[i][Of]
            psis[i][t] = 0
        else:   # 迭代更新deltas和psis的值
            deltas[i][t] = np.max(np.multiply([delta[t - 1] for delta in deltas],
                            [a[i] for a in A])) * B[i][Of]
            psis[i][t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas],
                            [a[i] for a in A]))

# 得到最优路径的终结点
I[0][T - 1] = np.argmax([delta[T - 1] for delta in deltas])

# 递归由后向前得到其他结点
for t in range(T - 2, -1, -1):
    I[0][t] = psis[int(I[0][t + 1])][t + 1]

# 输出最优路径
print('4. 最优路径是：', "————>".join([str(int(i + 1)) for i in I[0]]))

