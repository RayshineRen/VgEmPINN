## 1.Baseline

### 1.1 PINN

无调参$\lambda$

### 1.2 gPINN

w_x = w_y = 0.001

无调参$\lambda$

### 1.3 VPINN

无调参$\lambda$

### 1.4 hp-VPINN

NEx = NEy = 2

无调参$\lambda$



## 2.Models

### 2.1 VgPINN

调参如下

```python
wb = 20
wv = 1
wr = 1
```

trick: 可变参数，不收敛。

### 2.2 EmPINN

E: extended 扩维 $[x, y, x^{2}, y^{2}]$

m: modified 网络结构类似于ResNet

损失函数的权重参数可训练，定义如下：

```python
self.wb = 200 * tf.Variable(tf.ones([1], dtype=tf.float32), dtype=tf.float32)
self.wr = 10 * tf.Variable(tf.ones([1], dtype=tf.float32), dtype=tf.float32)
```

### 2.3 EmVgPINN

E: extended 扩维 $[x, y, x^{2}, y^{2}]$

m: modified 网络结构类似于ResNet

V: VPINN

g: gPINN w_x = w_y = 0.001

```python
self.wb = 200 * tf.Variable(tf.ones([1], dtype=tf.float64), dtype=tf.float64)
self.wr = 10 * tf.Variable(tf.ones([1], dtype=tf.float64), dtype=tf.float64)
self.wv = 10 * tf.Variable(tf.ones([1], dtype=tf.float64), dtype=tf.float64)
```

