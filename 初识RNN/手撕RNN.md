# 碎碎语

1. RNN之所以在自然语言处理领域用的多，是因为它有记忆。

2. 写工程时专门写一个文件存放工具函数，叫utils

3. 一层循环神经网络可以被看作一个单元的重复。所以，宽度对应重复次数，深度对应多少层。（三层其实就很多了）

4. tanh和sigmoid之间的区别和联系:

   (1) 函数上两者线性相关。

   ![1563004807283](/home/lixiang/.config/Typora/typora-user-images/1563004807283.png)

​                (2)sigmoid有个问题：**输出总是正数！！！**,初始化不好**优化路径容易出现zigzag现象**

​               （3）tanh过原点，而sigmoid不过，而且因为tanh在原点附近与y=x函数形式相近，所以当激活值较低时，可以直接进行矩阵运算，训练相对容易。

  		5. 在吴恩达课后作业里，见到了一个很神奇的写法： W * 分别与 a,x 矩阵乘 = Wc * concat(a,x)

# 反向传播

RNN:

> 1. 交叉熵损失求导 ：https://blog.csdn.net/jasonzzj/article/details/52017438
>
> 2. RNN反向传播：https://www.cnblogs.com/pinard/p/6509630.html



​      由于我们是基于时间反向传播，所以RNN的反向传播有时也叫做BPTT(back-propagation through time)。当然这里的BPTT和DNN也有很大的不同点，即这里所有的U,W,V,b,cU,W,V,b,c在序列的各个位置是共享的，反向传播时我们更新的是相同的参数。

   吴恩达课后作业代码实现和自己看博客的理论求解做法很不一样，硬是没看懂吴恩达课后作业的答案，没看到损失计算，最奇妙的是反向传播循环更新竟然是正向...个人感觉不该反向吗...

LSTM:

> https://www.cnblogs.com/pinard/p/6519110.html

 这里个人没有仔细推导，只是简单看了思路。





# tensorflow函数理解

> 1.https://blog.csdn.net/wjc1182511338/article/details/79689409
>
> 2.https://www.cnblogs.com/wzdLY/p/10071962.html



















​      



