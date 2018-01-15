
# coding: utf-8

# In[6]:


# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
seed = 9999


# # 欢迎来到线性回归项目
# 
# 若项目中的题目有困难没完成也没关系，我们鼓励你带着问题提交项目，评审人会给予你诸多帮助。
# 
# 所有选做题都可以不做，不影响项目通过。如果你做了，那么项目评审会帮你批改，也会因为选做部分做错而判定为不通过。
# 
# 其中非代码题可以提交手写后扫描的 pdf 文件，或使用 Latex 在文档中直接回答。

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[1]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何NumPy以及相关的科学计算库来完成作业


# 本项目要求矩阵统一使用二维列表表示，如下：
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

# 向量也用二维列表表示
C = [[1],
     [2],
     [3]]

#TODO 创建一个 4*4 单位矩阵
I = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]


# ## 1.2 返回矩阵的行数和列数

# In[2]:


# TODO 返回矩阵的行数和列数
def shape(M):
    matrix = list(M)
    if not matrix:
        return None
    else:
        return len(matrix),len(matrix[0])


# In[7]:


# 运行以下代码测试你的 shape 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 每个元素四舍五入到特定小数数位

# In[8]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    if not M:
        return None  
    
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j] = round(M[i][j], decPts)
    return None


# In[9]:


# 运行以下代码测试你的 matxRound 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 计算矩阵的转置

# In[10]:


# TODO 计算矩阵的转置
def transpose(M):  
    if not M:
        return None
    """recommended solution by reviewer"""
    return [list(col) for col in zip(*M)]



# In[11]:


# 运行以下代码测试你的 transpose 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 计算矩阵乘法 AB

# In[12]:


# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):     
    """recommended solution by reviewer"""
    _, c = shape(A)
    r, _ = shape(B)
    if c != r :
        raise ValueError
        
    Bt = transpose(B)
    result = [[sum((a*b) for a,b in zip(row,col)) for col in Bt] for row in A]
    return result


# In[13]:


# 运行以下代码测试你的 matxMultiply 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[14]:


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    if not A or not b or len(A) != len(b):
        return None
    """recommended solution by reviewer"""
    return [ra + rb for ra,rb in zip(A,b)]


# In[15]:


# 运行以下代码测试你的 augmentMatrix 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[16]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    if not M or r1 < 0 or r1 >= len(M) or r2 < 0 or r2 >= len(M):
        return None

    M[r1], M[r2] = M[r2], M[r1]
    
    pass


# In[17]:


# 运行以下代码测试你的 swapRows 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[18]:


# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if not M or r < 0 or r >= len(M) or scale == 0:
        raise ValueError
        
    for j in range(len(M[0])):
        M[r][j] *= scale
        
    pass


# In[19]:


# 运行以下代码测试你的 scaleRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[20]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    if not M or r1 < 0 or r2 < 0 or r1 >= len(M) or r2 >= len(M):
        raise ValueError
        
    for j in range(len(M[0])):
        M[r1][j] += M[r2][j] * scale
        
    pass


# In[21]:


# 运行以下代码测试你的 addScaledRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 2.3.1 算法
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# **注：** 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# ### 2.3.2 算法推演
# 
# 为了充分了解Gaussian Jordan消元法的计算流程，请根据Gaussian Jordan消元法，分别手动推演矩阵A为***可逆矩阵***，矩阵A为***奇异矩阵***两种情况。

# #### 推演示例 
# 
# 
# $Ab = \begin{bmatrix}
#     -7 & 5 & -1 & 1\\
#     1 & -3 & -8 & 1\\
#     -10 & -2 & 9 & 1\end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & \frac{1}{5} & -\frac{9}{10} & -\frac{1}{10}\\
#     0 & -\frac{16}{5} & -\frac{71}{10} & \frac{11}{10}\\
#     0 & \frac{32}{5} & -\frac{73}{10} & \frac{3}{10}\end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & 0 & -\frac{43}{64} & -\frac{7}{64}\\
#     0 & 1 & -\frac{73}{64} & \frac{3}{64}\\
#     0 & 0 & -\frac{43}{4} & \frac{5}{4}\end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & 0 & 0 & -\frac{3}{16}\\
#     0 & 1 & 0 & -\frac{59}{688}\\
#     0 & 0 & 1 & -\frac{5}{43}\end{bmatrix}$
#     
# 
# #### 推演有以下要求:
# 1. 展示每一列的消元结果, 比如3*3的矩阵, 需要写三步
# 2. 用分数来表示
# 3. 分数不能再约分
# 4. 我们已经给出了latex的语法,你只要把零改成你要的数字(或分数)即可
# 5. 检查你的答案, 可以用[这个](http://www.math.odu.edu/~bogacki/cgi-bin/lat.cgi?c=sys), 或者后面通过单元测试后的`gj_Solve`
# 
# _你可以用python的 [fractions](https://docs.python.org/2/library/fractions.html) 模块辅助你的约分_

# #### 以下开始你的尝试吧!

# In[89]:


# 不要修改这里！
from helper import *
A = generateMatrix(3,seed,singular=False)
b = np.ones(shape=(3,1),dtype=int) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
printInMatrixFormat(Ab,padding=3,truncating=0)


# 请按照算法的步骤3，逐步推演***可逆矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵。
# 
# 要求：
# 1. 做分数运算
# 2. 使用`\frac{n}{m}`来渲染分数，如下：
#  - $\frac{n}{m}$
#  - $-\frac{a}{b}$
# 
# 
# $ Ab = \begin{bmatrix}
#     7 & 5 & 3 & 1 \\
#     -5 & -4 & 6 & 1 \\
#     2 & -2 & -9 & 1 \end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & \frac{5}{7} & \frac{3}{7} & \frac{1}{7} \\
#     0 & -\frac{3}{7} & \frac{57}{7} & \frac{12}{7} \\
#     0 & -\frac{24}{7} & -\frac{69}{7} & \frac{5}{7} \end{bmatrix}$
# 
# $ --> $
# $\begin{bmatrix}
#     1 & 0 & \frac{3}{7} & \frac{8}{5} \\
#     0 & 1 & 0 & -\frac{53}{75} \\
#     0 & 0 & 1 & \frac{13}{75} \end{bmatrix}$
# 
# $ --> $ 
# $\begin{bmatrix}
#     1 & 0 & 0 & \frac{43}{75} \\
#     0 & 1 & 0 & -\frac{53}{75} \\
#     0 & 0 & 1 & \frac{13}{75} \end{bmatrix}$
#     
# 故此方程的解为
# $\begin{bmatrix}
#      \frac{43}{75} \\
#     -\frac{53}{75} \\
#     \frac{13}{75} \end{bmatrix}$
#     
# 

# In[90]:


# 不要修改这里！
A = generateMatrix(3,seed,singular=True)
b = np.ones(shape=(3,1),dtype=int)
Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
printInMatrixFormat(Ab,padding=3,truncating=0)


# 请按照算法的步骤3，逐步推演***奇异矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵。
# 
# 要求：
# 1. 做分数运算
# 2. 使用`\frac{n}{m}`来渲染分数，如下：
#  - $\frac{n}{m}$
#  - $-\frac{a}{b}$
# 
# 
# $ Ab = \begin{bmatrix}
#     -1 & 6 & -8 & 1 \\
#     -10 & -5 & 5 & 1 \\
#     -9 & 2 & -4 & 1 \end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     1 & -6 & 8 & -1 \\
#     0 & -65 & 85 & -9 \\
#     0 & -52 & 68 & -8 \end{bmatrix}$
#     
# $ --> \begin{bmatrix}
#     1 & 0 & \frac{2}{13} & 0 \\
#     0 & 1 & -\frac{17}{13} & 0 \\
#     0 & 0 & 0 & 1 \end{bmatrix}$
#     
# 因为第三行为前两行的线性组合，故此3阶方阵为奇异矩阵。
# 

# ### 2.3.3 实现 Gaussian Jordan 消元法

# In[22]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b
#from decimal import Decimal as D

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
    ================= recommended solution by reviewer ================
    def gj_Solve(A,b,decPts=4,epsilon=1.0e-16):
    if len(A) != len(b):
        raise ValueError

    Ab = augmentMatrix(A,b)

    for c in range(len(A[0])):
        AbT = transpose(Ab)
        col = AbT[c]
        maxValue = max(col[c:],key=abs)
        if abs(maxValue) < epsilon:
            return None

        maxIndex = col[c:].index(maxValue)+c

        swapRows(Ab,c,maxIndex)
        scaleRow(Ab,c,1.0/Ab[c][c])

        for i in range(len(A)):
            if Ab[i][c] != 0 and i != c:
                addScaledRow(Ab,i,c,-Ab[i][c])

    matxRound(Ab)

    return [[value] for value in transpose(Ab)[-1]]
"""
def is_near_zero(deci_value, epsilon):
    """check the decimal value is close to the threshold or not"""
    return abs(deci_value) < epsilon
    
def arrange_mat_pivot(A, row, col):
    """pick the largest abs value from lower triangle to make pivot element"""
    """changed according to the comments by reviewer"""
    row_for_swap, pivot_value, row_swap_found = row, A[row][col], False
    for rr in range(row + 1, len(A)):       
        if abs(pivot_value) < abs(A[rr][col]):
            row_for_swap, pivot_value, row_swap_found = rr, A[rr][col], True
    if row_swap_found:
        swapRows(A, row, row_for_swap)
                                                   
def transform_rref_triangle(pivot, A, decPts):
    """lower zero triangle"""
    for rr in range(pivot + 1, len(A)):   
        addScaledRow(A, rr, pivot, round((-1.0) * A[rr][pivot], decPts)) 
            
    """upper zero triangle"""
    for rr in range(0, pivot):   
        addScaledRow(A, rr, pivot, round((-1.0) * A[rr][pivot], decPts))

def fetch_resolved_vector(A):
    """construct the final b vector"""
    final_b = []
    for row in range(len(A)):
        cow_list = []
        cow_list.append(A[row][len(A)])
        final_b.append(cow_list)
    return final_b
    
def gj_Solve(A, b, decPts = 10, epsilon = 1.0e-16):
    """for decPts, we should ensure at lease 10 or above, otherwise we might expericed the bad result!"""
    if not A or not b or len(A) != len(A[0]) or len(b) != len(A):
        return None 
    
    aug_mat = augmentMatrix(A, b)
    if not aug_mat:
        return None
    
    matxRound(aug_mat, decPts)
    for row in range(len(aug_mat)):
        """check the pivot is zero or not, if zero then MUST be a singluar matrix and return None """
        arrange_mat_pivot(aug_mat, row, row)
        if is_near_zero(aug_mat[row][row], epsilon):
            return None
        """make the piviot into identiy element"""
        scaleRow(aug_mat, row, round(1.0/aug_mat[row][row], decPts))
        """make the lower and upper zero triangle"""
        transform_rref_triangle(row, aug_mat, decPts)  

    return fetch_resolved_vector(aug_mat)


# In[23]:


# 运行以下代码测试你的 gj_Solve 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## (选做) 2.4 算法正确判断了奇异矩阵：
# 
# 在算法的步骤3 中，如果发现某一列对角线和对角线以下所有元素都为0，那么则断定这个矩阵为奇异矩阵。
# 
# 我们用正式的语言描述这个命题，并证明为真。
# 
# 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合
# 

# TODO 证明：
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix}　$，将Ａ方阵按照第一列作为行列式展开，因为Ｉ为单位矩阵，Ｚ为全０矩阵，故Ａ的行列式展开的值等于Ｙ矩阵的行列式的值，又因为Ｙ的第一列全０，故Ｙ的行列式展开后的值一定为０.所以Ａ的行列式展开的值为０.故Ａ方阵一定是奇异矩阵。
# 

# # 3  线性回归

# ## 3.1 随机生成样本点

# In[24]:


# 不要修改这里！
# 运行一次就够了！
from helper import *
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

X,Y = generatePoints(seed,num=100)

## 可视化
plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')
plt.show()


# ## 3.2 拟合一条直线
# 
# ### 3.2.1 猜测一条直线

# In[26]:


#TODO 请选择最适合的直线 y = mx + b
m1 = 3
b1 = 7

# 不要修改这里！
plt.xlim((-5,5))
x_vals = plt.axes().get_xlim()
y_vals = [m1*x+b1 for x in x_vals]
plt.plot(x_vals, y_vals, '-', color='r')

plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')

plt.show()


# ### 3.2.2 计算平均平方误差 (MSE)

# 我们要编程计算所选直线的平均平方误差(MSE), 即数据集中每个点到直线的Y方向距离的平方的平均数，表达式如下：
# $$
# MSE = \frac{1}{n}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$

# In[27]:


# TODO 实现以下函数并输出所选直线的MSE
import math

def calculateMSE(X,Y,m,b):
    if not X or not Y or len(X) != len(Y):
        return 0
    
    total = 0.0   
    for ele in range(len(X)):
        total += math.pow((Y[ele] - m * X[ele] - b), 2)
    return total / len(X)

print(calculateMSE(X,Y,m1,b1))


# ### 3.2.3 调整参数 $m, b$ 来获得最小的平方平均误差
# 
# 你可以调整3.2.1中的参数 $m1,b1$ 让蓝点均匀覆盖在红线周围，然后微调 $m1, b1$ 让MSE最小。

# ## 3.3 (选做) 找到参数 $m, b$ 使得平方平均误差最小
# 
# **这一部分需要简单的微积分知识(  $ (x^2)' = 2x $ )。因为这是一个线性代数项目，所以设为选做。**
# 
# 刚刚我们手动调节参数，尝试找到最小的平方平均误差。下面我们要精确得求解 $m, b$ 使得平方平均误差最小。
# 
# 定义目标函数 $E$ 为
# $$
# E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 因为 $E = \frac{n}{2}MSE$, 所以 $E$ 取到最小值时，$MSE$ 也取到最小值。要找到 $E$ 的最小值，即要找到 $m, b$ 使得 $E$ 相对于 $m$, $E$ 相对于 $b$ 的偏导数等于0. 
# 
# 因此我们要解下面的方程组。
# 
# $$
# \begin{cases}
# \displaystyle
# \frac{\partial E}{\partial m} =0 \\
# \\
# \displaystyle
# \frac{\partial E}{\partial b} =0 \\
# \end{cases}
# $$
# 
# ### 3.3.1 计算目标函数相对于参数的导数
# 首先我们计算两个式子左边的值
# 
# 证明/计算：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# $$

# TODO 证明:
# 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# 推导上式：
# $$
# \frac{\partial E}{\partial m} = \frac{\partial E(m)}{\partial m} = \lim_{\Delta m\rightarrow0}\frac{E(m+\Delta m) -E(m)}{\Delta m} = \lim_{\Delta m\rightarrow0}\frac{\frac{1}{2}\sum_{i=1}^{n}{(y_i - (m + \Delta m)x_i - b)^2} - \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}}{\Delta m} = \frac{1}{2}\lim_{\Delta m\rightarrow0}\frac{\sum_{i=1}^{n}((2*(y_i - mx_i - b) - \Delta m)*(-1)*\Delta m)}{\Delta m} =  \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}    (推导完毕）
# $$
# 
# 
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# $$
# 
# 推导上式：
# $$
# \frac{\partial E}{\partial b} = \frac{\partial E(b)}{\partial b} = \lim_{\Delta b\rightarrow0}\frac{E(b+\Delta b) -E(b)}{\Delta b} = \lim_{\Delta b\rightarrow0}\frac{\frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - (b+\Delta b))^2} - \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}}{\Delta b} = \frac{1}{2}\lim_{\Delta b\rightarrow0}\frac{\sum_{i=1}^{n}((2*(y_i - mx_i - b) - \Delta b)*(-1)*\Delta b)}{\Delta b} =  \sum_{i=1}^{n}{-(y_i - mx_i - b)}    （推导完毕）
# $$
# 

# ### 3.3.2 实例推演
# 
# 现在我们有了一个二元二次方程组
# 
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$
# 
# 为了加强理解，我们用一个实际例子演练。
# 
# 我们要用三个点 $(1,1), (2,2), (3,2)$ 来拟合一条直线 y = m*x + b, 请写出
# 
# - 目标函数 $E$, 
# - 二元二次方程组，
# - 并求解最优参数 $m, b$

# TODO 写出目标函数，方程组和最优参数
# 
# 
# 1.目标函数：
# 
# 最终点目标函数：
# $$
# E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 此时点目标函数在求导数之后转化为：
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$
# 
# 2.二元二次方程组：
# 把三个点带入，则得到二元二次方程组：
# $$
# （1）-1×(1 -m -b) + -2*(2 - 2m - b) + -3*(2 - 3m - b) = 0 \\
#  (2) -(1 - m -b) + -(2 - 2m -b) + -(2 -3m -b) = 0
# $$
# 
# $$
# \begin{cases}
# \displaystyle
# 14m + 6b = 11 \\
# \\
# \displaystyle
# 6m + 3b = 5 \\
# \end{cases}
# $$
# 
# 
# 3.求解最优打m,b 
# $$
# m=\frac{1}{2}, b=\frac{2}{3}
# $$
# 

# ### 3.3.3 将方程组写成矩阵形式
# 
# 我们的二元二次方程组可以用更简洁的矩阵形式表达，将方程组写成矩阵形式更有利于我们使用 Gaussian Jordan 消元法求解。
# 
# 请证明 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = X^TXh - X^TY
# $$
# 
# 其中向量 $Y$, 矩阵 $X$ 和 向量 $h$ 分别为 :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$
# 
# 

# TODO 证明:
# $$
# X^TXh - X^TY = X^T(Xh-Y) = \begin{bmatrix}
#     x_1 & x_2... & x_n \\
#     1 & 1 ... & 1 
# \end{bmatrix}(\begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 
# \end{bmatrix} \begin{bmatrix}
#     m \\
#     b 
# \end{bmatrix} - \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n 
# \end{bmatrix}) =  \begin{bmatrix}
#     x_1 & x_2 ... & x_n \\
#     1 & 1 ... & 1 \\
#     \end{bmatrix} (\begin{bmatrix}
#     x_1*m + b \\
#     x_2*m + b \\
#     ... \\
#     x_n*m + b 
# \end{bmatrix} - \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n 
# \end{bmatrix}) = \begin{bmatrix}
#     x_1 & x_2... & x_n \\
#     1 & 1 ... & 1 \\
#     \end{bmatrix} \begin{bmatrix}
#     x_1 * m + b - y_1 \\
#     x_2 * m + b - y_2 \\
#     ... \\
#     x_n * m + b - y_n \\
# \end{bmatrix} = \begin{bmatrix}
#     \sum_{i=1}^{n}{-x_i(y_i - m * x_i - b)} \\
#     \sum_{i=1}^{n}{-(y_i - m * x_i - b)} \\
#     \end{bmatrix} = 
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} \\
# \end{bmatrix} 
# $$

# 至此我们知道，通过求解方程 $X^TXh = X^TY$ 来找到最优参数。这个方程十分重要，他有一个名字叫做 **Normal Equation**，也有直观的几何意义。你可以在 [子空间投影](http://open.163.com/movie/2010/11/J/U/M6V0BQC4M_M6V2AJLJU.html) 和 [投影矩阵与最小二乘](http://open.163.com/movie/2010/11/P/U/M6V0BQC4M_M6V2AOJPU.html) 看到更多关于这个方程的内容。

# ### 3.4 求解 $X^TXh = X^TY$ 
# 
# 在3.3 中，我们知道线性回归问题等价于求解 $X^TXh = X^TY$ (如果你选择不做3.3，就勇敢的相信吧，哈哈)

# In[28]:


# TODO 实现线性回归
import math
import time
'''
参数：X, Y 存储着一一对应的横坐标与纵坐标的两个一维数组
返回：m，b 浮点数
'''
def linearRegression(X,Y):
    if not X or not Y or len(X) != len(Y):
        return None, None
    """recommended solution by reviewer"""
    X = [[x,1] for x in X]
    Y = [[y] for y in Y]
    Xt = transpose(X)
    XtX = matxMultiply(Xt, X)
    XtY = matxMultiply(Xt, Y)
    h = gj_Solve(XtX,XtY)
    return h[0][0], h[1][0]

m2,b2 = linearRegression(X,Y)
assert isinstance(m2,float),"m is not a float"
assert isinstance(b2,float),"b is not a float"
print(m2,b2)


# 你求得的回归结果是什么？
# 请使用运行以下代码将它画出来。

# In[29]:


# 请不要修改下面的代码
x1,x2 = -5,5
y1,y2 = x1*m2+b2, x2*m2+b2

plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')
plt.plot((x1,x2),(y1,y2),'r')
plt.title('y = {m:.4f}x + {b:.4f}'.format(m=m2,b=b2))
plt.show()


# 你求得的回归结果对当前数据集的MSE是多少？

# In[30]:


print(calculateMSE(X,Y,m2,b2))

