# 前言

   本篇不会详细讲解RNN的知识，假设阅读者有了理论基础，想用tensorflow框架搭建自己的RNN,本篇主要目的是弄清楚tensorflow内构建RNN的方法,详细解读每个函数的参数和理论的联系。由于文章连贯性，代码会分开讲解。

**强推**

>   如果你是新手，对tesnroflow搭建RNN没有初步认识，强烈建议您去拜读这篇文章：
>
> ​	https://zhuanlan.zhihu.com/p/28196873
>
>   本人在写本文时找到此文，顿时觉得自己写的是比不上的，但都写了一半了，还是坚持写完。

# 输入数据

 此次数据集采用 mnist数据集合

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
minist = input_data.read_data_sets('MNIST_data',one_hot=True)
```

  以上面的数据集构建batch

```python
# batch_size = 50  #没批次50个样本
bactch_xs,batch_ys = minist.train.next_batch(batch_size)
```

这些数据如何传入tensroflow模型，入口构建(tensorflow函数的基本知识不再讲解)

```python
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
```

 最后，我向模型里输入的一个batch的数据大小维度是[50,784] （此形状后面还会改变）

# RNN搭建

> ​     根据tensorflow的特点，前向传播：搭建模型 ; 后向传播：tensorflow框架自动执行,所以，搭建RNN，无非是自己确定RNN的形状。

​       在代码实现上，我们用的时候都是用的RNNCell的两个子类BasicRNNCell和BasicLSTMCell。顾名思义，前者是RNN的基础类，后者是LSTM的基础类，这里以LSTM为例子。

```python
def RNN(X,weights,biases): 
# LSTM参数问题
# https://blog.csdn.net/wjc1182511338/article/details/79689409
# input = [batch_size,max_time,n_inputs]
    
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    # lstm_size, 个人理解为，a,c这些隐藏单元权重的的维度
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
    
    # https://www.cnblogs.com/wzdLY/p/10071962.html
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)   
    results = tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results
```

## 代码分条解读

> 第一行：数据维度转换

```python
# X本身的维度是[50,784]
# max_time = 28   # 时间步代表着单层RNN模型的宽度
# n_inputs = 28   # 一行有28个数据
# -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算
inputs = tf.reshape(X,[-1,max_time,n_inputs])
```

> 至此模型输入的一个batch的数据形状变为: [50,28,28]

---

> 第二行：构建一个基本LSTM单元（这个单元随着时间步子循环，便是一层LSTM网络）

```python
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
```

> **主要参数含义**：
>
> 1. **num_units**:  state_size = lstm_size,即理论里[a0,a1....an]的维度。
> 2. **forget_bias**: LSTM门的忘记系数，如果等于1，就是不会忘记任何信息，如果等于0，就都忘记
> 3. **state_is_tuple**：默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示



___

> 第三行：调用函数，自动按照时间步循环更新，时间步由数据维度决定（见上）

```python
 outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
```

>输入没什么说的，重点是输出：
>
> 1.**outputs**
>
>​      它是time_steps步里所有的输出。它的形状为(batch_size, time_steps, cell.output_size)，但要注意，tensorflow为了简便，其实这里的输出不是softmax后的记过，而是直接将隐藏计算出的 a（符号而已）直接输出，没有经过softmax层。
>
>2.**final_state**
>
>   state是最后一步的隐状态，它的形状为(batch_size, cell.state_size)
>
>   由于是lstm，它有两个隐态：C和a所以，这是一个元组。



___

>第四行，人工构造输出

```
results = tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
```

> 由于tensorflow输出并非实际softmax出处，所以需要人工处理。



> 剩下的代码无非就是：计算损失，确定优化方式，循环训练等，并非使用RNN独有



