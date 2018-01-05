---
typora-copy-images-to: ../MarkdownImages
---

#Machine Learning - Summary

the Andrew Ng course

##1. Linear Regression



## 2. Logistic Regression



## 3. Non-linear hypotheses

 ## 4. Neural Networks

ai(j)="activation" of unit i in layer j

Θ(j)=matrix of weights controlling function mapping from layer j to layer j+1

## 5. Neural Networks: Learning 		



- 神经网络，比logistic regression复杂


- 检查backpropagation是否正确（检查之后就不需要再进行检查了，因为计算很慢），使用gradient checking，gradApprox ： ![F36D2FF4-E32B-43DD-96DF-68B51F248675](/Users/mengke/Documents/DeepLearning/MarkdownImages/F36D2FF4-E32B-43DD-96DF-68B51F248675.png)

![3193591F-3ADB-48D5-900D-0D10FF8FE853](/var/folders/jf/sq0tj7jd6_173n1m9rmjf35r0000gn/T/abnerworks.Typora/3193591F-3ADB-48D5-900D-0D10FF8FE853.png)

- **Random Initialization:**

  ​如果全部初始化为0，这个网络变得对称symmetry，，权值不会更新

  ![8AF83ED8-D4D9-44BC-9DD9-2FA0176151DC](/var/folders/jf/sq0tj7jd6_173n1m9rmjf35r0000gn/T/abnerworks.Typora/8AF83ED8-D4D9-44BC-9DD9-2FA0176151DC.png)	

  ```Octave
  If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

  Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
  Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
  Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
  ```

  ​

- **Training a neural network：**

  设计网络结构：
  ![29F6ECD6-33B9-4B9D-B127-85C6E4DD8A82](/Users/mengke/Documents/DeepLearning/MarkdownImages/29F6ECD6-33B9-4B9D-B127-85C6E4DD8A82.png)

  ​

  训练步骤：
  ![62EB7C39-8740-4EF3-A0A7-D1AFA0BBC69E](/Users/mengke/Documents/DeepLearning/MarkdownImages/62EB7C39-8740-4EF3-A0A7-D1AFA0BBC69E.png)




## 6. Neural Networks: 
































