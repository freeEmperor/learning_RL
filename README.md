# README

​		（半成品）

​		我已经进行了相关理论算法的学习，正在尝试实现gym_taxi项目。



## Taxi_policyIteration

​		这是我利用朴素的 model based 的策略迭代算法实现的项目。

#### 		理论：

​		我们随意构造policy $\pi_0$，反复使用policy_evaluation和policy_improvement得到条( policy - state_value )链：
$$
\pi_0 \to v_{\pi_0} \to \pi_1 \to v_{\pi_1} \to \ ...\ \to \pi_k \to v_{\pi_k} \to  \pi_{k+1} \to  \ ...
$$
直至policy $\pi$ 收敛。

​		**policy_evaluation：**当已有policy $\pi_k$ 时，以此计算state_value $v_{\pi_k}$ 的值，随意取初始state_value $v_{\pi_k}^{(0)}$，利用公式迭代：
$$
v_{\pi_k}^{(t+1)}=f(v_{\pi_k}^{(t)})= r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}^{(t)}
$$

直至 $v_{\pi_k}$ 收敛。

​		**policy_iteration：**根据计算出的state_value $v_{\pi_k}$ ，我们依据公式改进policy $\pi_k$ 为 $\pi_{k+1}$：
$$
\pi_{k+1} = \arg\max_\pi (\ r_\pi + \gamma P_\pi v_{\pi_k} \ )
$$
具体的，有：
$$
a^*(s) = \arg\max_a (\ r(a|s)+ \gamma v_{\pi_k}(s') \ ) \ , \  \ \ s\stackrel{a}{\longrightarrow} s'
\\
\pi_{k+1}(a|s)=[a=a^*(s)]
$$

#### 	Taxi_policyIteration_getPolicy.py：

​		该代码对上述算法进行了实践。我选取discount_rate $\gamma = 0.9$ ，对策略迭代了500次，认为 $\pi_{500}$ 是最终收敛的policy。 其核心代码为：

```python
get_model() #对于model_based的算法，预处理出一个字典，记录每个(state, acton)pair会来到的新状态
for i in range(0, 500): #迭代500次
    get_P() #处理出P矩阵
    get_R() #处理出r向量
    iteration_state_value() #用迭代法计算state_value
    update_policy() #更新policy
```

#### 	Policy.csv

​		Taxi_policyIteration_getPolicy.py将最终得到的policy保存为Policy.csv文件方便检验时读取。

#### 	Taxi_policyIteration_test.py：

​		该代码使用gym接口得到了gym-taxi项目的环境。我们随机设置一个初始状态，可视化地渲染每一帧，可以观察到出租车的行为，从而判断我们的策略是否足够优。
