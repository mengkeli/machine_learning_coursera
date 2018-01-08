---
typora-copy-images-to: ../MarkdownImages
---

#Machine Learning - Summary

*——the Andrew Ng course*

##1. Linear Regression

### 1. 分类问题 / 回归问题

分类：目标是推出一组离散的结果

回归：通过回归来推出一个连续的输出



###2. 无监督学习 / 监督学习	

Unsupervised Learning：给算法大量的数据(无标签)，并让算法为我们从数据中找出某种结构。


垃圾邮件问题、新闻事件分类

Supervised Learning：有标签	

​		

### 3. 单变量线性回归 

只含有一个特征/输入变量：

​			

### 4. 代价函数 Cost Function

也称为：平方误差函数（还有其他类型的代价函数）

![FA1F962B-8932-44FF-AA64-D95BE1BCE02C](https://ws2.sinaimg.cn/large/006tNc79gy1fn810nzv5ij30jx03674h.jpg)

​	![85ECED30-D3DF-4ECD-B136-C2E0D3DF9E62](https://ws4.sinaimg.cn/large/006tNc79gy1fn80yx5itpj30i60aajt3.jpg)	



### 5. 梯度下降 Gradient Descent

![B47AC7BB-3034-441A-9E76-F04F24177F6E](https://ws3.sinaimg.cn/large/006tNc79gy1fn80z0ci3bj309q08et93.jpg)

目标：最小化损失函数

方法：用梯度下降的方式一步一步接近 J 函数最低点

![BCF6B849-8C0D-48FD-A989-82DCF69D1B89](https://ws2.sinaimg.cn/large/006tNc79gy1fn810j1rjlj30x00h876c.jpg)

![6449D94D-7CDF-4707-A986-489CA0D6DF69](https://ws3.sinaimg.cn/large/006tNc79gy1fn810gw09zj30n90d2409.jpg)

下右图每一个圆圈上的 J 值相同：

![33E6E81F-61C8-4CCF-AA8A-98F80D645D93](https://ws1.sinaimg.cn/large/006tNc79gy1fn810q0ltdj30w00gm41i.jpg)

Note：

- 批量（Batch）梯度下降中，我们每一次都**同时**让所有的参数减去学习速率乘以代价函数的导数。		
  “Batch”: Each step of gradient descent uses all the training examples.

-  a 是**学习率**(learning rate)，它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大。

- 随着梯度下降法的运行，梯度移动的幅度会自动变得越来越小，直到最终移动幅度非常小。所以没有必要另外再减小a，保持a不变就可以了。

  ![3AD57FFC-5C1F-4192-8F19-E9AE22E63293](https://ws2.sinaimg.cn/large/006tNc79gy1fn810kuhq7j31200l6djt.jpg)



### 6. 用梯度下降实现线性回归

模型表示：

![6B1A542A-B893-4300-A6F0-97212F20369B](https://ws3.sinaimg.cn/large/006tNc79gy1fn80yuj8xzj30u80aqaba.jpg)

梯度表示：

![1515225089672](https://ws2.sinaimg.cn/large/006tNc79gy1fn80yqen4gj30qa0eaaby.jpg)

正规方程也可以求解 J 最小值：

- 不需要选择a 学习率
- 一次运算得出参数
- 矩阵逆运算代价大，但是在数据量较大时n>10000，运算很复杂
- 只适用于线性模型，不适合逻辑回归等其他模型

### 7. 多变量线性回归
![1515225848650](https://ws1.sinaimg.cn/large/006tNc79gy1fn80ytafeej30ok0dyjt8.jpg)


### 8. 多变量梯度下降

![1515225941334](https://ws3.sinaimg.cn/large/006tNc79gy1fn8108yxc4j30wa0hkae2.jpg)

### 9. 特征缩放 Feature Scaling

![1515226123300](https://ws1.sinaimg.cn/large/006tNc79gy1fn80ys2ytfj30fi080t8w.jpg)

### 10. 选择学习率

通常考虑a = 0.01，0.01， 0.1， 0.3， 1， 3 ， 10



## 2. Logistic Regression

###1. 逻辑回归模型表示

![1515226441383](https://ws1.sinaimg.cn/large/006tNc79gy1fn810i6m4yj30wu0hw775.jpg)

代价函数表示为：

![F24DAA27-4251-4530-801B-07723A598D78](https://ws3.sinaimg.cn/large/006tNc79gy1fn8108w7d0j30wa0hkae2.jpg)

![D0025B50-CFAB-4EA3-A9DB-8557DEF5EC20](https://ws3.sinaimg.cn/large/006tNc79gy1fn80yyesq7j30xc096ab9.jpg)

### 2. 优化算法

- conjugate gradient 共轭梯度法
- BFGS 局部优化法
- L-BFGS 

优点：不需要手动选择学习率a， 比梯度下降快。缺点：更复杂



### 3. 多类别分类：一对多

将一个类标为正类，其余为负类，得到多个分类器。

n个类别，要训练n(n-1)/2 分类器，输入x，选择概率最大的那个输出



### 4. 正则化 Regularization

出现过拟合，如何处理：

1. 丢弃多余特征。
   - 手工选择保留哪些特征
   - 采用模型选择的方法（PCA等）
2. 正则化。
   - 保留所有特征，但是减少参数的大小


模型中的高次项导致了过拟合的产生，如果能够让高次项的系数接近0的话，就能很好防止过拟合。

选取合理lambda的值，如果lambda太大，为了使得cost function尽可能小，所有theta值都会减小，最后模型接近于直线，拟合程度差。



### 5. 正则化线性回归

cost function表示：

![](https://ws3.sinaimg.cn/large/006tNc79gy1fn80wb1s0tj30dd031mxg.jpg)

正则化线性回归的梯度下降算法，核心在于，在原有算法的更新规则上，另theta额外减少了一个值：



Note：

- theta0不参与正则化，它是bias单元
- ​

###6. 正则化逻辑回归

![1515308231157](../MarkdownImages/1515308231157.jpg)

![1515308206577](../MarkdownImages/1515308206577.jpg)



 ## 3. Neural Networks

### 1. 模型表示

![1515308690226](../MarkdownImages/1515308690226.jpg)

Note：

- 每一个a都是由上一层的所有x和每一个x对应的theta决定的
- theta(j)代表从j层映射到j+1层的权重矩阵，尺寸为j+1层激活单元数量为行数，j层激活单元数量为列数
- ![1515308875009](../MarkdownImages/1515308875009.jpg)
- theta · X = a，X的特征是以列的形式排列的

### 2. 网络结构理解

![1515310059299](../MarkdownImages/1515310059299.jpg)

其实神经网络就像是 logistic regression，只不过我们把 logistic regression 中的输入向量[x1 ~ x3]变成了中间层的 [a1 ~ a3], 即:

我们可以把 a0 , a1 , a2 , a3 看成更为高级的特征值，也就是 x0 , x1 , x2 , x3 的进化体，并且它们是由 x 与 theta决定的，因为是梯度下降的，所以 a 是变化的，并且变得越来越厉害，所以这些更高级的特征值远比仅仅将 x 次方厉害，也能更好的预测新数据。

**这就是神经网络相比于逻辑回归和线性回归的优势。**

非线性分类示例：XOR/XNOR, AND

![1515310402797](../MarkdownImages/1515310402797.jpg)



### 3. 多类分类

输出结果用one-hot vector表示



###4. Cost Funcion

标记方法：![1BBD5FEF-62E5-492C-8159-31BAD75BB299](https://ws2.sinaimg.cn/large/006tNc79gy1fn811yakd8j30au02mmxb.jpg)

logistic regression 的cost function：

![09283CC1-4621-4D68-B0D5-9623E2BCB7FD](https://ws2.sinaimg.cn/large/006tNc79gy1fn80z2zjv1j30yu02974p.jpg)

neural networks 的cost function：

![2F6FD407-446D-457E-A7FB-FE26AE927A23](https://ws3.sinaimg.cn/large/006tNc79gy1fn810mzzwzj317k046aaw.jpg)

Note:

- the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
- the triple sum simply adds up the squares of all the individual Θs in the entire network.
- the i in the triple sum does **not** refer to training example i（i指的是下一层神经元的下标，j指的是本层神经元的下标）
- 对每一行特征，给出K个预测，选取概率最大的一个。




###5. 反向传播算法 Backpropagation Algorithm

正向传播：我们从第一层开始正向一层一层进行计算，直到最后一层的 h(x)			![3A1EC5A4-0072-497C-AA11-6A81DE29F27B](https://ws3.sinaimg.cn/large/006tNc79gy1fn810c0ectj30d8074mxy.jpg)

反向传播：也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。

![62C0F19B-5B3A-4240-8B7D-1F341C6B16D6](https://ws3.sinaimg.cn/large/006tNc79gy1fn80ymsucvj30om0d9q5c.jpg)

直观理解：

![772304D4-AB8E-4D92-9CC4-D32876AA8A99](https://ws3.sinaimg.cn/large/006tNc79gy1fn810e89foj30k00b9ace.jpg)



###6. 梯度检验 Gradent checking

梯度的数值检验（Numerical Gradient Checking），检查backpropagation是否正确（检查之后就不需要再进行检查了，因为计算很慢）

gradApprox ：             ![F36D2FF4-E32B-43DD-96DF-68B51F248675](https://ws2.sinaimg.cn/large/006tNc79gy1fn80z1gjgkj306v01z747.jpg)



###7. Random Initialization

如果我们令所有的初始参数都为 0，这将意味着我们第二层的所有激活单元都会有相同的值。这个网络变得对称symmetry，，权值不会更新

![8AF83ED8-D4D9-44BC-9DD9-2FA0176151DC](/var/folders/jf/sq0tj7jd6_173n1m9rmjf35r0000gn/T/abnerworks.Typora/8AF83ED8-D4D9-44BC-9DD9-2FA0176151DC.png)	

```Octave
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```



###8. Training a neural network

设计网络结构：

![8AEEB51A-8425-4202-BCD6-5AC044768DA6](https://ws2.sinaimg.cn/large/006tNc79gy1fn80yp6tzij30zu089wg5.jpg)

训练步骤：

![EDCEC152-1C66-40F1-80E2-31EDA21857C9](https://ws1.sinaimg.cn/large/006tNc79gy1fn810b4b7aj30yy08udhs.jpg)

1. 参数随机初始化


2. 正向传播计算所有h(x)
3. 编写计算代价函数 J 的代码
4. 反向传播计算所有偏导数
5. 梯度数值检验（检验偏导数）
6. 使用优化算法来最小化代价函数

```
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```



## 4.  Evaluate a Learnign Algorithm 算法评估 

### 1. 检验算法是否过拟合

数据分为70%训练集、30%测试集，注意数据均衡，再随机打乱。

![1515317158368](../MarkdownImages/1515317158368.jpg)						

![1515317248000](../MarkdownImages/1515317248000.jpg)



### 2. 模型选择 Model Selection

用cross_validation，数据分为60%训练集、20%交叉验证集、20%测试集

模型选择的方法为:

（假设我们要在 10 个不同次数的二项式模型之间进行选择）

1. 使用训练集训练出 10 个模型

2. 用 10 个模型分别对交叉验证集计算得出交叉验证误差(代价函数的值)

3. 选取代价函数值最小的模型

4. 用步骤 3 中选出的模型对测试集计算得出推广误差(代价函数的值)

   ​

### 3. Bias vs. Variance 偏差和方差

高偏差和高方差的问题基本上来说是欠拟合和过拟合的问题：		



将训练集和交叉验证集的代价函数误差与多项式的次数绘制在同一张图表上：



Note：

- 训练集误差和交叉验证集误差近似时: 偏差/欠拟合
- 交叉验证集误差远大于训练集误差时: 方差/过拟合

我们选择一系列的想要测试的 λ 值，通常是 0-10 之间的呈现 2 倍关系的值(如:0,0.01,0.02,0.04,0.08,0.15,0.32,0.64,1.28,2.56,5.12,10 共 12 个)。


选择 λ 的方法为:

1. 使用训练集训练出 12 个不同程度正则化的模型
2. 用 12 模型分别对交叉验证集计算的出交叉验证误差
3. 选择得出交叉验证误差最小的模型
4. 运用步骤 3 中选出模型对测试集计算得出推广误差，我们也可以同时将训练集和交叉验证集模型的代价函数误差与 λ 的值绘制在一张图表上

• 当 λ 较小时，训练集误差较小(过拟合)而交叉验证集误差较大
• 随着 λ 的增加，训练集误差不断增加(欠拟合)，而交叉验证集误差则是先减小后 增加



 ### 4. Learning Curves

使用学习曲线来判断某一个学习算法是否处于偏差、方差问题。	


学习曲线是学习算法的一个很好的合理检验(sanity check)。



​		


Note：

- 在高偏差/欠拟合的情况下，增加数据到训练集不一定能有帮助
- 高方差/过拟合的情况下，增加更多数据到训练集可能可以提高算法效果。
- ​

**神经网络的偏差和方差**：

- 通常选择较大的神经网络并采用正则化处理会比采用较小的神经网络效果要好。

- 对于神经网络中的隐藏层的层数的选择，通常从一层开始逐渐增加层数（模型选择）

  ​

### 5. Error Analysis 误差分析

构建一个学习算法的推荐方法为:

1. 从一个简单的能快速实现的算法开始，实现该算法并用交叉验证集数据测试这个算法
2. 绘制学习曲线，决定是增加更多数据，或者添加更多特征，或者其他选择
  3. 进行误差分析：人工检查**交叉验证集**中我们算法中产生预测误差的实例，看看这些实例是否有某种系统化的趋势。通过量化的数值评估误差。


偏斜类（skewed classes）: 某种类特别多，其他类少

查准率Precision和查全率Recall = TP / (TP + FN)：



Precision = TP / (TP + FP) : 	在所有我们预测有恶性肿瘤的病人中，实际上有恶性肿
瘤的病人的百分比，越高越好。


Recall = TP / (TP + FN) : 在所有实际上有恶性肿瘤的病人中，成功预测有恶性肿瘤
的病人的百分比，越高越好。

accurac = TP + FP / total samples

F1 score = 2 * P * R / (P + R): 查全率与查准率之间的权衡




## 5. Support Vector Machine





## 6. Dimensionlity Reduction



## 7. Recommendar Systems



## 8. Large Scale Machine Learning






















