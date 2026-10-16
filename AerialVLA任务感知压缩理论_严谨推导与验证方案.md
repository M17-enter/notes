# AerialVLA 任务感知压缩：可认证、边端可计算的率失真下界

> 版本：2026-10-15  
> 文档性质：对既有讨论的严格化整理。本文明确区分已经证明的结论、需要附加假设才能成立的统计证书，以及尚属工程映射的部分。除特别说明外，对数均为自然对数，信息量单位为 nat；除以 $\ln 2$ 后得到 bit。

## 摘要

本文研究如下问题：在无人机边端观察完整视觉状态、云端执行 AerialVLA 决策，且导航指令等信息由边云双方共享时，为了使压缩后的云端动作分布与完整 AerialVLA 的动作分布足够接近，至少必须传递多少任务相关信息？

严格结论可以概括为：定义教师动作后验

$$
\Pi_F(X,U)=\pi_F(\cdot\mid X,U),
$$

并采用任务失真

$$
d_T(X,q\mid U)=D_{\mathrm{KL}}\!\left(\Pi_F(X,U)\|q\right),
$$

则相应的条件任务率失真函数为

$$
R_T(D)=\inf_{P(Q\mid X,U):\,\mathbb E d_T(X,Q\mid U)\le D}I(X;Q\mid U).
$$

若 $S=\phi(X,U)$ 是边端可计算的统计量，$\lambda\ge 0$，并且函数 $\alpha(S,U)$ 对几乎所有 $u$ 满足

$$
\sup_{q}\mathbb E\!\left[
e^{\alpha(S,u)-\lambda d_T(X,q\mid u)}\mid U=u
\right]\le 1,
$$

那么有严格下界

$$
\boxed{R_T(D)\ge \mathbb E[\alpha(S,U)]-\lambda D.}
$$

进一步，如果通过独立校准数据认证上述对偶可行性，并用独立测试数据得到 $\mathbb E\alpha$ 的置信下界 $\underline\mu_\alpha$，则以预定置信度有

$$
\boxed{R_T(D)\ge [\underline\mu_\alpha-\lambda D]_+.}
$$

这就是目前最合适的“有信息论依据、可在 AerialVLA 边端在线计算”的证据下界。其边端在线阶段只需计算 $S_t$ 和已离线认证的 $\alpha(S_t,U_t)$，不需要运行完整 AerialVLA 教师。

但必须准确陈述其含义：它首先是**总体平均任务信息率下界**。单帧的 $\alpha_t$ 不能自动解释为“这一帧至少需要多少 bit”，也不能未经额外论证直接除以单个视觉 token 的大小得到逐帧最小 $K_t$。

---

## 1. AerialVLA 场景与随机变量

AerialVLA 使用前视图和下视图组成双视图输入，以 OpenVLA-7B 为骨干，将视觉表示与包含目标描述、模糊方向提示的语言提示共同输入模型，并自回归产生三个数值动作 token。论文将三个动作维度分别离散为 99 个类别，并设置 LAND/近零位移终止机制。原模型是基于当前观测和即时提示的反应式策略。这些结构事实决定了本文可将任务输出定义为有限离散动作变量，也可先研究其确定性粗粒度版本。

定义：

$$
\begin{aligned}
X_t &: \text{时刻 }t\text{ 边端独有的完整观测；}\\
U_t &: \text{边云双方在通信前都已知的共享信息；}\\
A_t^F &: \text{由完整 AerialVLA 教师诱导的离散动作变量；}\\
S_t&=\phi(X_t,U_t):\text{边端能够低成本计算的统计量；}\\
M_t &: \text{边端发送的消息；}\\
Q_t &: \text{云端依据 }(M_t,U_t)\text{形成的动作概率分布。}
\end{aligned}
$$

必须遵守如下记账原则：

1. 只有编码前云端确实知道的信息才能放入 $U$。若历史图像、私有状态或边端预测结果只在无人机端存在，它们应属于 $X$，不能当成免费侧信息。
2. 如果云端另有共享历史、地图或状态估计，也应纳入 $U$。遗漏真正的云端侧信息可能高估通信需求。
3. 本文首先对单时刻记号推导。若序列存在依赖，应把整个块或历史纳入 $X$，或者明确采用条件/有记忆信源模型；不能把同一轨迹中的帧未经处理地当作独立同分布样本。

### 1.1 教师任务，而非真实 Bayes 专家

现实中通常无法知道真实的最优专家后验 $P(A\mid X,U)$。因此严格做法不是声称完整 AerialVLA 就是真实 Bayes 专家，而是定义教师随机变量

$$
A^F\mid X=x,U=u\sim \pi_F(\cdot\mid x,u).
$$

于是

$$
\Pi_F(x,u)=P(A^F=\cdot\mid X=x,U=u)=\pi_F(\cdot\mid x,u)
$$

是该教师任务的精确后验。本文下界严格回答：

> 为了保留完整 AerialVLA 教师的概率决策信息，至少需要多少条件通信信息？

若要进一步把它解释为真实导航专家的信息下界，还需要独立证明或实验证明教师与专家任务之间的关系。

---

## 2. 为什么 KL 是严格的任务失真

云端输出任意动作分布 $q$。对给定 $(x,u)$，完整教师的最小期望 log-loss 为

$$
\mathbb E_{A^F\sim\pi_F}[-\log \pi_F(A^F\mid x,u)]
=H(\pi_F(\cdot\mid x,u)).
$$

若云端只能使用 $q$，期望 log-loss 为

$$
\mathbb E_{A^F\sim\pi_F}[-\log q(A^F)].
$$

二者之差恰好为

$$
\boxed{
d_T(x,q\mid u)
=D_{\mathrm{KL}}\!\left(\pi_F(\cdot\mid x,u)\|q\right).
}
$$

因此 KL 不是任意选取的经验距离，而是相对于完整教师后验所增加的期望概率决策 log-loss。

若 $M$ 是压缩消息，云端使用真实条件后验 $P(A^F\mid M,U)$ 作为最优 soft decoder，则

$$
\begin{aligned}
&H(A^F\mid M,U)-H(A^F\mid X,U)\\
&\qquad=I(A^F;X\mid M,U)\\
&\qquad=\mathbb E D_{\mathrm{KL}}\!\left(
P(A^F\mid X,U)\|P(A^F\mid M,U)
\right),
\end{aligned}
$$

其中使用了给定 $(X,U)$ 后 $A^F$ 与 $M$ 条件独立的编码马尔可夫关系。这是 log-loss、远程源编码与 Information Bottleneck 之间的精确连接。

---

## 3. 条件任务率失真函数

定义

$$
\boxed{
R_T(D)=
\inf_{P(Q\mid X,U):\,\mathbb E d_T(X,Q\mid U)\le D}
I(X;Q\mid U).
}
\tag{1}
$$

这里 $Q$ 是理论上的 reproduction variable，即云端最终形成的动作概率分布，并不要求它等于实际发送的视觉 token。

### 命题 1：动作后验是任务充分统计量

令 $\Pi=\Pi_F(X,U)$。因为失真只通过 $\Pi$ 依赖 $X$，有

$$
\boxed{
R_T(D)=
\inf_{P(Q\mid \Pi,U):\,\mathbb E D_{\mathrm{KL}}(\Pi\|Q)\le D}
I(\Pi;Q\mid U).
}
\tag{2}
$$

**证明。** 对任意 $P(Q\mid X,U)$，对具有相同 $(\Pi,U)$ 的 $X$ 做条件平均，得到信道 $P(Q\mid\Pi,U)$。这一操作保持 $(\Pi,Q,U)$ 的联合分布，因此保持平均失真；同时由数据处理不等式

$$
I(\Pi;Q\mid U)\le I(X;Q\mid U).
$$

反向不等式由直接选择只依赖 $(\Pi,U)$ 的信道得到，故等式成立。□

这说明真正需要保留的不是视觉 $X$ 的全部信息，而是视觉中影响教师动作后验 $\Pi$ 的部分，正是 Information Bottleneck 的“保留相关变量信息”思想在本问题中的率失真实现。

### 命题 2：实际通信量的操作性下界

设实际系统由边端产生消息 $M$，云端由 $(M,U)$ 产生 $Q$，并满足平均任务失真不超过 $D$。则

$$
\boxed{I(X;M\mid U)\ge R_T(D).}
\tag{3}
$$

**证明。** 给定 $U$ 后有马尔可夫链 $X\to M\to Q$，故

$$
I(X;M\mid U)\ge I(X;Q\mid U)\ge R_T(D).
$$

□

若对每个共享侧信息 $u$，消息使用条件前缀码或其他可唯一解码的二进制码，码长为 $L$，则

$$
\mathbb E L\ge H_2(M\mid U)\ge I_2(X;M\mid U),
$$

因此

$$
\boxed{
\mathbb E L\ge \frac{R_T(D)}{\ln 2}\quad\text{bit}.
}
\tag{4}
$$

---

## 4. 核心定理：边端可计算的 Csiszár 对偶证据下界

### 定理 1：Edge-Measurable Dual Witness

取任意 $\lambda\ge0$，令 $S=\phi(X,U)$ 为边端可计算统计量。若可测函数 $\alpha(S,U)$ 对 $P_U$-几乎所有 $u$ 满足

$$
\boxed{
\sup_{q\in\mathcal Q}
\mathbb E\!\left[
e^{\alpha(S,u)-\lambda d_T(X,q\mid u)}
\mid U=u
\right]\le1,
}
\tag{5}
$$

则对所有 $D\ge0$，

$$
\boxed{
R_T(D)\ge\mathbb E[\alpha(S,U)]-\lambda D.
}
\tag{6}
$$

这里 $\mathcal Q$ 是动作概率单纯形；若动作被粗粒度化，它就是相应较低维单纯形。

**证明。** 任取满足平均失真约束的测试信道 $P(Q\mid X,U)$。固定 $(q,u)$，相对熵的变分不等式给出

$$
\begin{aligned}
&D_{\mathrm{KL}}(P_{X\mid q,u}\|P_{X\mid u})
+\lambda\mathbb E[d_T(X,q\mid u)\mid q,u]\\
&\quad\ge
\mathbb E[\alpha(S,u)\mid q,u]
-\log\mathbb E_{X\mid u}
e^{\alpha(S,u)-\lambda d_T(X,q\mid u)}.
\end{aligned}
$$

由式 (5)，最后的对数项不大于 0，故右侧不小于 $\mathbb E[\alpha(S,u)\mid q,u]$。再对 $(Q,U)$ 求期望，得到

$$
I(X;Q\mid U)+\lambda\mathbb E d_T(X,Q\mid U)
\ge\mathbb E\alpha(S,U).
$$

因为 $\mathbb E d_T\le D$ 且 $\lambda\ge0$，

$$
I(X;Q\mid U)\ge\mathbb E\alpha(S,U)-\lambda D.
$$

对所有可行测试信道取下确界即得式 (6)。□

### 4.1 任意候选评分的严格正规化

对任意候选函数 $f(S,U)$，定义

$$
C_{\lambda,f}(u)=
\sup_q\mathbb E\!left[
e^{f(S,u)-\lambda d_T(X,q\mid u)}\mid U=u
\right].
$$

只要该量有限，令

$$
\boxed{
\alpha_{\lambda,f}(S,u)=f(S,u)-\log C_{\lambda,f}(u),
}
\tag{7}
$$

就自动满足式 (5)。因此神经网络或其他边端评分器只有在完成这一全局正规化或等价可行性认证后，才是严格的对偶 witness；未经认证的预测分数本身不是率失真下界。

### 4.2 一个解析可行解与 $I-D$ 下界

令

$$
\bar\pi_u=\mathbb E[\Pi\mid U=u],\qquad
\alpha(\Pi,u)=D_{\mathrm{KL}}(\Pi\|\bar\pi_u),qquad \lambda=1.
$$

由 Jensen 不等式可验证

$$
\mathbb E_{\Pi\mid u}
e^{D_{\mathrm{KL}}(\Pi\|\bar\pi_u)-D_{\mathrm{KL}}(\Pi\|q)}
\le1,
$$

故

$$
\boxed{
R_T(D)\ge I(A^F;X\mid U)-D.
}
\tag{8}
$$

这里使用恒等式

$$
\mathbb E_{X\mid u}D_{\mathrm{KL}}(\Pi\|r_u)
=I(A^F;X\mid U=u)+D_{\mathrm{KL}}(\bar\pi_u\|r_u).
\tag{9}
$$

式 (9) 中是**加号**。因此若用不准确的参考先验 $r_u$ 代替真正的边缘后验 $\bar\pi_u$，平均 KL 会高估真实的条件任务互信息；此前讨论中出现过的减号写法是错误的。

式 (8) 是重要的解析基线，但在线计算它需要获得或近似教师后验 $\Pi$。最终边端方案仍应使用经过离线认证、只依赖轻量统计量 $S$ 的 $\alpha(S,U)$。

---

## 5. 与 $D$-tilted information 和逐样本 local rate 的关系

在满足通常正则条件、且存在率失真最优 reproduction 分布 $P^*_{Q\mid U}$ 时，令 $\lambda^*=-R_T'(D)\ge0$。条件 $D$-tilted information 可写为

$$
\jmath_T(x,D\mid u)
=-\log\mathbb E_{Q^*\mid u}
\exp\{\lambda^*D-\lambda^*d_T(x,Q\mid u)\}.
\tag{10}
$$

若定义

$$
b^*(x,u)=-\log\mathbb E_{Q^*\mid u}
e^{-\lambda^*d_T(x,Q\mid u)},
$$

则

$$
\jmath_T(x,D\mid u)=b^*(x,u)-\lambda^*D,
\qquad
\mathbb E\jmath_T(X,D\mid U)=R_T(D).
$$

这解释了为什么对偶函数具有逐样本意义。但它依赖未知的真实分布和最优 reproduction 分布，不能直接在边端获得。本文的 $\alpha(S,U)$ 是对这一最优对偶对象施加“边端可测”约束后得到的合法 witness，通常更松，但可部署。

在对应的最优测试信道

$$
P^*(dq\mid x,u)
=\frac{P^*(dq\mid u)e^{-\lambda^*d_T(x,q\mid u)}}
{\int P^*(dq'\mid u)e^{-\lambda^*d_T(x,q'\mid u)}}
$$

下，定义局部平均失真

$$
\delta^*(x,u)=\mathbb E[d_T(x,Q\mid u)\mid x,u]
$$

和局部率贡献

$$
r^*(x,u)=D_{\mathrm{KL}}(P^*_{Q\mid x,u}\|P^*_{Q\mid u}).
$$

直接代入可得

$$
\boxed{r^*(x,u)=b^*(x,u)-\lambda^*\delta^*(x,u),}
\tag{11}
$$

且 $\mathbb E r^*(X,U)=R_T(D)$。这比单独的 $b^*$ 更接近逐样本的“local rate contribution”。但是只有在获得最优 primal-dual 对及测试信道时，式 (11) 才成立。一般的可行 $\alpha(S,U)$ 没有配套的最优测试信道，不能直接被称为逐样本最小码率。

因此当前最严谨的措辞是：

- $\mathbb E\alpha-\lambda D$ 是平均任务率的严格下界；
- $\alpha_t$ 是构成该平均证书的边端可计算对偶贡献或任务难度分数；
- 只有最优条件下的 $r_t^*$ 才具有精确的局部互信息贡献含义；
- 从 $\alpha_t$ 或 $r_t^*$ 到实际动态 token 数 $K_t$ 仍需额外的预算分配或架构约束论证。

---

## 6. 有限状态下的可计算构造

为使边端计算极轻量，可把 $S$ 离散为有限状态 $S\in\{1,\ldots,J\}$。先固定一个可重复采样的条件任务 $U=u$，定义

$$
w_j(u)=P(S=j\mid U=u)
$$

以及

$$
k_{\lambda,j}(q;u)=
\mathbb E\!\left[
e^{-\lambda D_{\mathrm{KL}}(\Pi_F\|q)}
\mid S=j,U=u
\right].
\tag{12}
$$

令 $t_j(u)=e^{\alpha(j,u)}$。对偶可行性等价于

$$
\sum_{j=1}^Jw_j(u)t_j(u)k_{\lambda,j}(q;u)\le1,
\qquad\forall q\in\Delta_m.
\tag{13}
$$

可以求解

$$
\boxed{
\begin{aligned}
\max_{t_1,\ldots,t_J}\quad&
\sum_{j=1}^Jw_j\log t_j\\
\text{s.t.}\quad&
\sum_{j=1}^Jw_jt_jk_{\lambda,j}(q)\le1,
\quad\forall q\in\Delta_m,\\
&1\le t_j\le e^{A_0}.
\end{aligned}}
\tag{14}
$$

上下界 $1\le t_j\le e^{A_0}$ 不是理论必需条件，而是一个保守的可选限制：它确保 $0\le\alpha\le A_0$，方便数值优化和有限样本浓缩界。因为 $e^{-\lambda d}\le1$，$t_j=1$ 总是可行，所以该受限问题至少有一个可行点。允许负 $\alpha$ 可能得到更紧的目标，但统计范围也需相应调整。

式 (14) 是半无限凸优化：目标是凹函数，固定 $q$ 时约束对 $t$ 线性。可用 cutting-plane 方法反复寻找违反最严重的 $q$，再加入约束。

对于 KL 失真，

$$
e^{-\lambda D_{\mathrm{KL}}(\pi\|q)}
=e^{\lambda H(\pi)}\prod_{a=1}^m q_a^{\lambda\pi_a}.
$$

当 $0<\lambda\le1$ 时，该函数关于 $q$ 为凹函数，有限正权混合仍为凹函数，因此 cutting-plane 的分离子问题

$$
\sup_{q\in\Delta_m}\sum_jw_jt_jk_{\lambda,j}(q)
$$

可以作为标准凸优化求得全局最优。若 $\lambda>1$，该凹性一般不再保证，需要全局优化、分支定界或带可证明覆盖误差的网格证书，不能只依靠局部最大值。

### 6.1 动作粗粒度化是合法的较松下界

完整联合动作可能有 $99^3$ 个组合并带 LAND。可令 $B=g(A^F)$ 为 LAND/non-LAND、粗方向或其他确定性动作映射，并把 $\Pi_B$ 和 $Q_B$ 定义为后验的 push-forward。KL 数据处理不等式给出

$$
D_{\mathrm{KL}}(\Pi_A\|Q_A)
\ge D_{\mathrm{KL}}(\Pi_B\|Q_B).
$$

同时确定性映射不会增加互信息。因此在相同侧信息和相应 KL 失真约束下，

$$
\boxed{R_A(D)\ge R_B(D).}
\tag{15}
$$

所以先对二元或低维粗动作任务建立证书，确实能给完整动作任务提供合法但可能较松的下界。

---

## 7. 最大识别性难点：完整指令 $U$ 不能被随意粗化

对偶条件要求对几乎所有完整共享指令 $u$ 控制

$$
P(X\mid U=u).
$$

若有限数据中每条完整自然语言指令只出现一次，经验联合分布会让 $U$ 成为样本标识：给定 $U=u$ 即确定唯一 $X=x_u$ 及教师后验。在该经验分布上，云端仅凭 $U$ 就能记住后验，因而

$$
R_T^{\mathrm{emp}}(D)=0,
\qquad D\ge0.
$$

这不是优化失败，而是条件分布不可识别。

尤其不能在没有假设时把完整指令 $U$ 替换为方向类别 $V$。云端拥有更多侧信息通常可以降低所需码率，因此对较粗侧信息得到的 $R(D\mid V)$ 下界，不一定是原问题 $R(D\mid U)$ 的下界。只有在下列情况之一成立时，粗化才严谨：

1. 研究问题本身被明确重新定义为双方只共享 $V$；
2. 有可验证的条件不变性，例如 $P(X,\Pi,S\mid U)=P(X,\Pi,S\mid V)$；
3. 建立了从粗化问题到完整侧信息问题的额外比较定理。

推荐的严格解决方案是定义可重复采样的条件任务分布：固定完整目标描述和提示 $u$，在模拟器中使用独立环境、初始状态、随机种子和轨迹生成多个条件样本

$$
X_{u,1},\ldots,X_{u,n_u}\overset{\mathrm{iid}}{\sim}P(X\mid U=u).
$$

如果样本来自轨迹，应以独立 episode 为统计单位，或使用块抽样/混合过程界；同一轨迹中的大量相邻帧不能被当作同样数量的独立样本。

若模拟器分布与真实部署分布不同，得到的严格结论首先只适用于所定义的模拟器任务分布。推广到真实环境需要分布迁移、密度比、有界分布偏移或真实部署样本上的再次认证。

---

## 8. 有限样本统计认证

仅在训练样本上使经验约束成立，不足以证明总体对偶可行。建议使用 episode 级三路拆分：

- 训练集：选择 $S$、学习分区或评分器并优化候选 $\widehat\alpha$；
- 校准集：对冻结的 $\widehat\alpha$ 认证对偶约束；
- 测试集：对最终认证后的 $\alpha_{\mathrm{cert}}$ 估计总体均值的下置信界。

### 8.1 对偶可行性的校准

定义

$$
C_{\widehat\alpha}(u)=
\sup_q\mathbb E\!\left[
e^{\widehat\alpha(S,u)-\lambda d_T(X,q\mid u)}
\mid U=u
\right].
$$

若独立校准过程以概率至少 $1-\delta_{\mathrm{cal}}$ 给出同时上界

$$
\overline C(u)\ge C_{\widehat\alpha}(u)
\quad\text{对所有待认证 }u,
$$

则令

$$
\boxed{
\alpha_{\mathrm{cert}}(S,u)
=\widehat\alpha(S,u)-\log\overline C(u)
}
\tag{16}
$$

即可在同一事件上保证总体对偶可行性。

若 $0\le\widehat\alpha\le A_0$，则对固定 $(u,q)$，随机变量

$$
Y=e^{\widehat\alpha(S,u)-\lambda d_T(X,q\mid u)}
$$

位于 $[0,e^{A_0}]$。因此可使用 Hoeffding、empirical Bernstein 或其他上置信界。要从有限候选 $q$ 推广到整个连续单纯形，必须再加上严格的覆盖误差或使用能够求总体上界的结构化方法。仅检查一个有限网格、却不控制网格间隙，不能证明式 (5)。

一种可行技术路线是对教师后验做预先声明的平滑，使各动作概率有统一正下界，再利用 $q\mapsto e^{-\lambda KL(\pi\|q)}$ 的连续/Hölder 模量和 $\varepsilon$-net 给出覆盖修正。此时证书严格对应“平滑后的教师任务”；若要返回原任务，还需另给平滑误差界。

### 8.2 $\mathbb E\alpha$ 的测试集下置信界

冻结 $\alpha_{\mathrm{cert}}$ 后，若独立测试单位共有 $N$ 个，且

$$
a_{\min}\le\alpha_{\mathrm{cert}}(S,U)\le a_{\max},
$$

记范围 $B=a_{\max}-a_{\min}$，测试均值为 $\widehat\mu_\alpha$。Hoeffding 给出

$$
\boxed{
\underline\mu_\alpha
=\widehat\mu_\alpha
-B\sqrt{\frac{\log(1/\delta_{\mathrm{eval}})}{2N}}.
}
\tag{17}
$$

于是用 union bound，以至少

$$
1-\delta_{\mathrm{cal}}-\delta_{\mathrm{eval}}
$$

的概率，

$$
\boxed{
R_T(D)\ge
[\underline\mu_\alpha-\lambda D]_+.
}
\tag{18}
$$

如果在多个 $\lambda$、多种动作粗化、多个评分器中看完测试结果后再挑最优者，式 (18) 的置信度会失效。应在训练/校准阶段完成选择后只在测试集评一次，或对所有候选使用同步置信界和多重比较修正。

---

## 9. “$\underline\mu_\alpha-\lambda D>0$”到底验证什么

当 $\lambda>0$ 时，非平凡条件等价于

$$
\boxed{D<\underline\mu_\alpha/\lambda.}
\tag{19}
$$

它验证的是：在预先定义的 AerialVLA 教师任务分布、动作变量、共享侧信息、失真阈值和置信水平下，任意达到该失真要求的通信系统都必须传递严格为正的条件任务信息。

这对应三个层次：

1. **命题一——理论正确性：** 定理 1 证明任何总体可行的 $\alpha$ 都产生率失真下界。
2. **命题二——边端可计算性：** 有限状态优化或轻量评分器使推理时只需计算 $S$ 和 $\alpha(S,U)$。
3. **命题三——统计可靠性：** 独立校准和测试把经验量变成给定置信度的总体证书，即式 (18)。

因此 $\underline\mu_\alpha-\lambda D>0$ 是命题三所给下界“非平凡”的判据，而不是动态 $K$ 已经被证明最优的判据。

建议预先选定一组具有任务意义的 $D$，或画认证曲线

$$
D\longmapsto [\underline\mu_\alpha-\lambda D]_+.
$$

若对多组 $\lambda$ 取上包络，应按前述方式处理模型选择和多重比较。稳定性还应报告不同 episode 划分、不同环境、seen/unseen 条件下的结果，但重复 bootstrap 本身不能替代对偶可行性的正式校准。

---

## 10. AerialVLA 上的推荐验证协议

### 10.1 明确任务对象

1. 固定教师 checkpoint、softmax 温度和动作提取方式。
2. 首先选低维粗动作 $B=g(A^F)$，例如 LAND/non-LAND 或粗方向；成功后再扩大动作字母表。
3. 明确 $X$ 包含前视/下视图、哪些状态和哪些历史；明确 $U$ 包含目标描述、模糊方向提示及云端已有信息。
4. 将实验结论表述为“对该 checkpoint 诱导的教师任务成立”。

### 10.2 建立可重复条件采样

1. 选择一组固定完整指令 $u$。
2. 对每个 $u$，用独立地图/种子/初始姿态生成多个 episode。
3. 以 episode 为单位划分 train/calibration/test，避免相邻帧泄漏。
4. 保存每个样本的 $(X,U,S,\Pi_F,\text{episode id})$。

### 10.3 求解并校准 witness

1. 对一组预先规定的 $\lambda\in(0,1]$ 构造有限状态 $S$。
2. 用训练集估计 $w_j,k_{\lambda,j}$，求解式 (14)。
3. 使用全局 separation oracle 检查最坏 $q$，不能只随机采样少量 $q$。
4. 在校准集计算 $\overline C(u)$，按式 (16) 下移评分。
5. 在完全未参与选择的测试集上计算式 (17) 和式 (18)。

### 10.4 必须同时报告的结果

- 认证下界曲线 $[\underline\mu_\alpha-\lambda D]_+/\ln2$，单位 bit/决策；
- 约束最坏点 $\sup_q$ 的校准上界及置信水平；
- 每个完整指令/任务条件的独立 episode 数；
- 评分范围和校准下移量 $\log\overline C(u)$；
- seen、unseen object、unseen map 分布上的独立结果；
- $\alpha_t$ 的分布、与障碍/转向/LAND 等语义事件的关系；
- 不同实际 $K$ 下的任务 KL、导航成功率、SPL 和真实通信 bit 数。

### 10.5 动态 $K$ 的额外实验命题

可令边端策略为

$$
K_t=\operatorname{clip}\bigl(
\lceil a\,\alpha(S_t,U_t)+b\rceil,
K_{\min},K_{\max}
\bigr),
$$

或在整条轨迹上按 $\alpha_t$ 分配固定总 token 预算。但这一步属于经验证的预算控制器，而不是由式 (18) 自动推出的逐帧最小 token 定理。需要另行比较：固定 $K$、随机 $K$、熵/置信度启发式、未经认证的学生分数及认证 $\alpha$ 策略。

如果每个传输 token 的实际码长为 $B_{\mathrm{tok}}$ bit，最多只能从平均通信界得到

$$
\mathbb E[K]B_{\mathrm{tok}}
\ge \frac{R_T(D)}{\ln2}
$$

这一平均必要条件。它通常会很松，因为 Shannon 理论允许远强于现有视觉 token compressor 的任意编码器。要获得对现有 6L Compressor 更有力量的 $K$ 下界，必须进一步研究架构受限率失真或限制编码器计算能力。

---

## 11. 已严格成立、附条件成立与尚未证明的结论

### 已严格成立

1. 教师后验 KL 等于相对完整教师的额外期望 log-loss。
2. 后验 $\Pi_F$ 是该 KL 任务的充分统计量，式 (1) 可化为式 (2)。
3. 实际消息的条件互信息及可唯一解码平均码长受 $R_T(D)$ 下界约束。
4. 任何满足式 (5) 的边端可测 $\alpha(S,U)$ 都给出式 (6)。
5. 动作确定性粗粒度化可给完整动作问题提供合法的较松下界。
6. 解析 witness 给出 $R_T(D)\ge I(A^F;X\mid U)-D$。

### 在明确条件下成立

1. 式 (18) 需要训练、校准和测试独立，且对连续 $q$ 的 supremum 得到真正的统一认证。
2. iid 浓缩界要求统计单位独立；轨迹数据需按 episode 处理或换用依赖数据理论。
3. 用方向类别代替完整指令只在任务重新定义或条件不变性成立时合法。
4. $D$-tilted information 和局部率公式需要存在适当的最优 primal-dual 解及常规正则条件。

### 尚未由当前理论证明

1. $\alpha_t/\ln2$ 是单帧必需 bit 数；
2. $\alpha_t$ 与最优 token 数 $K_t^*$ 成比例；
3. 将下界直接除以一个 FP16 视觉 token 的位数即可得到有用的逐帧 $K$ 下界；
4. 在模拟器上认证的下界无需额外假设即可迁移到真实飞行；
5. AerialVLA 教师任务下界自动等于真实专家或导航成功率的下界。

---

## 12. 最终结论

目前最稳固的理论链条是

$$
\boxed{
\text{教师 log-loss 任务}
\to \Pi_F
\to R_T(D)
\to \text{Csiszár 对偶}
\to \alpha(S,U)
\to \text{校准可行性}
\to [\underline\mu_\alpha-\lambda D]_+.
}
$$

它同时满足三点：

1. **理论上是真正的通信信息率下界**，而不是相关性启发式；
2. **在线时可在 AerialVLA 边端计算**，因为完整教师后验和全局对偶优化都被移到离线阶段；
3. **可以给出有限样本置信声明**，前提是完整条件任务可重复采样，并严格完成对连续 reproduction 空间的统一校准。

接下来最关键的实证问题正是：在预先定义的 AerialVLA 条件任务分布上，是否能稳定得到

$$
\boxed{\underline\mu_\alpha-\lambda D>0,}
$$

以及认证后的 $\alpha_t$ 是否对动态 token 分配具有额外预测价值。前者是严格理论证书的非平凡性，后者是需要单独验证的系统设计命题。

---

## 参考文献

1. Tishby, N., Pereira, F. C., & Bialek, W. *The Information Bottleneck Method*. arXiv:physics/0004057. https://arxiv.org/abs/physics/0004057
2. Courtade, T. A., & Weissman, T. *Multiterminal Source Coding under Logarithmic Loss*. IEEE Transactions on Information Theory, 60(1), 740–761, 2014. https://arxiv.org/abs/1110.3069
3. Kostina, V., & Verdú, S. *Fixed-Length Lossy Compression in the Finite Blocklength Regime*. IEEE Transactions on Information Theory, 58(6), 3309–3338, 2012. https://arxiv.org/abs/1102.3944
4. Yang, Y., & Mandt, S. *Towards Empirical Sandwich Bounds on the Rate-Distortion Function*. ICLR 2022. https://arxiv.org/abs/2111.12166
5. Xu, P., Deng, Z., Deng, J., Gu, Z., & Wan, S. *AerialVLA: A Vision-Language-Action Model for UAV Navigation via Minimalist End-to-End Control*. arXiv:2603.14363, 2026. https://arxiv.org/abs/2603.14363

## 记号速查

| 记号 | 含义 |
|---|---|
| $X$ | 边端独有的完整观测 |
| $U$ | 边云预先共享的侧信息 |
| $A^F$ | 完整 AerialVLA 教师诱导的动作变量 |
| $\Pi_F$ | 教师动作后验 $\pi_F(\cdot\mid X,U)$ |
| $S$ | 边端可计算统计量 $\phi(X,U)$ |
| $M$ | 实际通信消息 |
| $Q$ | 云端形成的动作概率分布 |
| $d_T$ | 教师后验与云端后验之间的 KL 失真 |
| $R_T(D)$ | 条件任务率失真函数 |
| $\lambda$ | 率失真对偶斜率参数 |
| $\alpha(S,U)$ | 经认证的边端可测对偶 witness |
| $\underline\mu_\alpha$ | $\mathbb E\alpha$ 的统计下置信界 |
| $K$ | 实际传输的视觉 token 数 |
