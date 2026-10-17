# AerialVLA：条件化 D-tilted Log-Partition 任务信息下界

> 版本：2026-10-16  
> 目的：检查并严格化“沿着 D-tilted information 推导边端任务信息下界”的新路线。  
> 约定：除特别说明外，对数为自然对数，信息单位为 nat；除以 $\ln 2$ 后得到 bit。公式使用 `$...$` 与 `$$...$$`，以提高 Markdown 阅读器兼容性。

## 0. 审核结论

新推导的核心方向是正确的。可以严格建立

$$
R_T(D)
\ge
\sup_{\lambda\ge 0}
\left\{
F_S^{\mathrm{tilt}}(\lambda)-\lambda D
\right\},
$$

其中

$$
F_S^{\mathrm{tilt}}(\lambda)
=
\inf_{\nu_{Q|U}}
\mathbb E_{S,U}
\left[
-\log\int k_\lambda(S,U;q)\,\nu(dq|U)
\right],
$$

而条件核为

$$
k_\lambda(s,u;q)
=
\mathbb E\left[
e^{-\lambda D_{\mathrm{KL}}(\Pi_F\|q)}
\mid S=s,U=u
\right].
$$

若最优重建边缘分布 $\nu_\lambda^*$ 存在，则

$$
\alpha_\lambda^*(s,u)
=
-\log\int k_\lambda(s,u;q)\,\nu_\lambda^*(dq|u)
$$

满足 Csiszár 型对偶约束

$$
\mathbb E\left[
e^{\alpha_\lambda^*(S,u)-\lambda D_{\mathrm{KL}}(\Pi_F\|q)}
\mid U=u
\right]
\le 1,
\qquad \forall q,
$$

从而

$$
\boxed{
R_T(D)
\ge
\mathbb E[\alpha_\lambda^*(S,U)]-\lambda D.
}
$$

这条路线的真正意义是：它不再先人为规定评分函数，而是从一个条件化 log-partition 优化中构造出**最优的 $S$-可测对偶 witness**。

但需要修正原稿中的四种过强表述：

1. 它不是与此前 Edge Dual 完全不同的新下界，而是同一个受限 Csiszár 对偶问题的 log-partition 原始形式；两者在适当条件下强对偶。
2. $\alpha_\lambda^*(s,u)$ 一般不是单帧最小码长，只是最优的逐样本率–失真 Lagrangian 贡献。
3. one-shot converse 需要明确固定大小码本、给定共享侧信息的编码模型及对偶约束的逐条件成立。
4. 训练样本上的有限 $q$ 集优化不是总体下界；最终仍需要全局 supremum 和独立数据校准。

以下给出完整推导。

---

## 1. 问题定义

设

$$
\begin{aligned}
H &: \text{边端拥有的完整视觉观测或视觉特征},\\
U &: \text{编码前边云双方都已知的完整导航条件},\\
S&=\phi(H,U):\text{边端可低成本计算的统计量},\\
A^F &: \text{完整 AerialVLA 教师诱导的离散动作变量},\\
\Pi_F(H,U)&=\pi_F(\cdot|H,U):\text{教师动作后验},\\
Q&:\text{云端形成的动作后验重建}.
\end{aligned}
$$

严格地说，定义教师任务

$$
A^F|H=h,U=u\sim\pi_F(\cdot|h,u).
$$

因此本文证明的是保持完整 AerialVLA 教师决策信息所需的通信下界，而不是未经验证地把 AerialVLA 等同于真实 Bayes 专家。

对云端后验 $q\in\Delta_m$，定义任务失真

$$
\boxed{
d(h,q|u)
=D_{\mathrm{KL}}\!\left(Pi_F(h,u)\|q\right).
}
\tag{1}
$$

该失真正好等于使用 $q$ 代替完整教师后验后增加的期望 log-loss。

条件任务率失真函数定义为

$$
\boxed{
R_T(D)
=
\inf_{P(Q|H,U):\,\mathbb E d(H,Q|U)\le D}
I(H;Q|U).
}
\tag{2}
$$

因为失真只通过 $\Pi_F(H,U)$ 依赖 $H$，可以把 $H$ 无损地约化为任务后验 $\Pi_F$：

$$
R_T(D)
=
\inf_{P(Q|\Pi_F,U):\,\mathbb E D_{\mathrm{KL}}(\Pi_F\|Q)\le D}
I(\Pi_F;Q|U).
\tag{3}
$$

式 (3) 来自条件平均和数据处理不等式。它说明该问题本质上是一个 conditional posterior rate-distortion problem。

### 基本正则条件

为了避免把技术条件隐藏在符号中，后文默认：

1. $H,S,U$ 所在空间为标准 Borel 空间，条件分布存在；
2. 动作字母表有限，因而 $q$ 位于紧概率单纯形 $\Delta_m$；
3. $D_{\mathrm{KL}}(\pi\|q)$ 允许取 $+\infty$，并约定 $e^{-\lambda\infty}=0$；
4. 所涉及 log-partition 在相关样本上为正且可积；
5. 当声称最优 $\nu_\lambda^*$ 存在时，假设核关于 $q$ 连续或至少具有保证弱下半连续性和紧性的条件。

有限动作空间和适当的教师后验平滑通常可使这些条件容易满足。

---

## 2. 率失真 Lagrangian

对任意 $\lambda\ge0$，定义

$$
\boxed{
F_H(\lambda)
=
\inf_{P(Q|H,U)}
\left{
I(H;Q|U)+\lambda\mathbb E d(H,Q|U)
\right\}.
}
\tag{4}
$$

任取满足 $\mathbb E d\le D$ 的测试信道，有

$$
I(H;Q|U)
\ge
F_H(\lambda)-\lambda\mathbb E d
\ge
F_H(\lambda)-\lambda D.
$$

故

$$
\boxed{
R_T(D)
\ge
F_H(\lambda)-\lambda D,
\qquad \forall\lambda\ge0.
}
\tag{5}
$$

从而

$$
R_T(D)
\ge
\sup_{\lambda\ge0}
\{F_H(\lambda)-\lambda D\}.
\tag{6}
$$

在经典凸性、闭性和可达条件下，式 (6) 取等号；本文构造下界时只需要“$\ge$”。

---

## 3. Gibbs 变分表示：完整证明

令 $\nu(dq|u)$ 是任意依赖共享侧信息 $u$ 的候选重建边缘分布。对任意测试信道 $P(Q|H,U)$，有恒等式

$$
\begin{aligned}
&\mathbb E D_{\mathrm{KL}}
\bigl(P_{Q|H,U}\|\nu_{Q|U}\bigr)\\
&\quad=
I(H;Q|U)
+\mathbb E D_{\mathrm{KL}}
\bigl(P_{Q|U}\|\nu_{Q|U}\bigr).
\end{aligned}
\tag{7}
$$

因此

$$
I(H;Q|U)
=
\inf_{\nu_{Q|U}}
\mathbb E D_{\mathrm{KL}}
\bigl(P_{Q|H,U}\|\nu_{Q|U}\bigr),
\tag{8}
$$

最优值在 $\nu=P_{Q|U}$ 处取得。

把式 (8) 代入式 (4)，并交换联合下确界的次序：

$$
\begin{aligned}
F_H(\lambda)
=\inf_{\nu_{Q|U}}
\inf_{P(Q|H,U)}
\mathbb E\bigl[
&D_{\mathrm{KL}}(P_{Q|H,U}\|\nu_{Q|U})\\
&+\lambda d(H,Q|U)
\bigr].
\end{aligned}
\tag{9}
$$

固定 $(h,u)$ 和 $\nu(\cdot|u)$。Gibbs 变分公式给出

$$
\inf_{P_{Q|h,u}}
\left{
D_{\mathrm{KL}}(P_{Q|h,u}\|\nu_{Q|u})
+\lambda\mathbb E[d(h,Q|u)|h,u]
\right}
=
-\log\int e^{-\lambda d(h,q|u)}\nu(dq|u).
\tag{10}
$$

当分母为正时，最优条件分布为

$$
P_\nu^*(dq|h,u)
=
\frac{
e^{-\lambda d(h,q|u)}\nu(dq|u)
}{
\int e^{-\lambda d(h,q'|u)}\nu(dq'|u)
}.
\tag{11}
$$

于是

$$
\boxed{
F_H(\lambda)
=
\inf_{\nu_{Q|U}}
\mathbb E_{H,U}
\left[
-\log
\int e^{-\lambda d(H,q|U)}\nu(dq|U)
\right].
}
\tag{12}
$$

式 (12) 是 Blahut–Arimoto/Gibbs log-partition 结构的条件版本。

---

## 4. 对边端统计量条件化：核心 Jensen 下界

定义条件指数核

$$
\boxed{
k_\lambda(s,u;q)
=
\mathbb E\left[
e^{-\lambda d(H,q|u)}
\mid S=s,U=u
\right].
}
\tag{13}
$$

对固定 $\nu$，记

$$
Z_\nu(h,u)
=
\int e^{-\lambda d(h,q|u)}\nu(dq|u),
$$

以及

$$
\bar Z_\nu(s,u)
=
\int k_\lambda(s,u;q)\nu(dq|u).
\tag{14}
$$

由 Tonelli/Fubini 定理和条件期望定义，

$$
\mathbb E[Z_\nu(H,U)|S=s,U=u]
=\bar Z_\nu(s,u).
\tag{15}
$$

因为 $x\mapsto-\log x$ 为凸函数，条件 Jensen 不等式给出

$$
\mathbb E[-\log Z_\nu(H,U)|S,U]
\ge
-\log\bar Z_\nu(S,U).
\tag{16}
$$

再取总期望并对同一个可行集合中的 $\nu$ 取下确界：

$$
\boxed{
F_H(\lambda)
\ge
F_S^{\mathrm{tilt}}(\lambda),
}
\tag{17}
$$

其中

$$
\boxed{
F_S^{\mathrm{tilt}}(\lambda)
=
\inf_{\nu_{Q|U}}
\mathbb E_{S,U}
\left[
-\log\bar Z_\nu(S,U)
\right].
}
\tag{18}
$$

结合式 (5)，得到：

### 定理 1：条件化 D-tilted 下界

对任意 $\lambda\ge0$，

$$
\boxed{
R_T(D)
\ge
F_S^{\mathrm{tilt}}(\lambda)-\lambda D.
}
\tag{19}
$$

从而

$$
\boxed{
R_T(D)
\ge
\sup_{\lambda\ge0}
\left{
F_S^{\mathrm{tilt}}(\lambda)-\lambda D
\right}.
}
\tag{20}
$$

这个不等式方向是正确的。关键是式 (16) 对每个固定 $\nu$ 都成立；因此对两边关于同一集合取 infimum 后，方向仍保持。

若 $S=H$，则条件期望退化，式 (16) 取等号，所以

$$
F_S^{\mathrm{tilt}}(\lambda)=F_H(\lambda).
$$

当 $S$ 比 $H$ 粗时，Jensen gap 只会使下界变松，不会虚假提高原始任务率失真下界。

### 有效失真的解释

当 $\lambda>0$ 时定义

$$
\widetilde d_\lambda(s,q|u)
=
-\frac1\lambda\log k_\lambda(s,u;q).
\tag{21}
$$

则

$$
F_S^{\mathrm{tilt}}(\lambda)
=
\inf_{\nu_{Q|U}}
\mathbb E
\left[
-\log\int
e^{-\lambda\widetilde d_\lambda(S,q|U)}
\nu(dq|U)
\right].
$$

因此它是一个由条件指数平均诱导的辅助率失真 Lagrangian。注意 $\widetilde d_\lambda$ 本身依赖 $\lambda$，不能把不同 $\lambda$ 下的辅助问题误认为同一个固定失真问题。

---

## 5. 最优边端评分及其与 Edge Dual 的严格等价

假设式 (18) 的最优解存在，记为 $\nu_\lambda^*(dq|u)$。定义

$$
Z_\lambda^*(s,u)
=
\int k_\lambda(s,u;q)\nu_\lambda^*(dq|u),
\tag{22}
$$

以及

$$
\boxed{
\alpha_\lambda^*(s,u)
=-\log Z_\lambda^*(s,u).
}
\tag{23}
$$

显然

$$
F_S^{\mathrm{tilt}}(\lambda)
=\mathbb E[\alpha_\lambda^*(S,U)].
\tag{24}
$$

因为 $0\le k_\lambda\le1$ 且 $\nu$ 是概率分布，所以 $0<Z_\lambda^*\le1$ 时 $\alpha_\lambda^*\ge0$。

### 5.1 一阶最优性条件

固定 $u$，考虑方向扰动

$$
\nu_\varepsilon
=(1-\varepsilon)\nu_\lambda^*+\varepsilon\delta_q,
\qquad \varepsilon\downarrow0.
$$

目标函数的右方向导数为

$$
\left.\frac{d}{d\varepsilon}
\mathbb E_{S|u}[-\log Z_{\nu_\varepsilon}(S,u)]
\right|_{0+}
=
1-
\mathbb E_{S|u}
\frac{k_\lambda(S,u;q)}{Z_\lambda^*(S,u)}.
\tag{25}
$$

最优性要求该导数对每个 $q$ 非负，因此

$$
\boxed{
\mathbb E_{S|u}
\frac{k_\lambda(S,u;q)}{Z_\lambda^*(S,u)}
\le1,
\qquad\forall q.
}
\tag{26}
$$

对 $\nu_\lambda^*(\cdot|u)$ 支持上的 $q$，在通常互补条件下取等号。

由 $e^{\alpha_\lambda^*}=1/Z_\lambda^*$ 和式 (13)，式 (26) 等价于

$$
\boxed{
\mathbb E\left[
e^{\alpha_\lambda^*(S,u)-\lambda d(H,q|u)}
\mid U=u
\right]
\le1,
\qquad\forall q.
}
\tag{27}
$$

这正是边端可测 Csiszár 对偶可行性条件。因此

$$
\boxed{
R_T(D)
\ge
\mathbb E[\alpha_\lambda^*(S,U)]-\lambda D.
}
\tag{28}
$$

### 5.2 它是最优的 $S$-可测对偶 witness

考虑对偶问题

$$
\begin{aligned}
\sup_{\alpha}\quad&\mathbb E[\alpha(S,U)]\\
\text{s.t.}\quad&
\mathbb E\left[
e^{\alpha(S,u)-\lambda d(H,q|u)}|U=u
\right]\le1,\\
&\hspace{7.2cm}\forall q, P_U\text{-a.e. }u.
\end{aligned}
\tag{29}
$$

任取可行 $\alpha$ 和任意 $\nu$，先把约束关于 $q$ 对 $\nu(dq|u)$ 积分，可得

$$
\mathbb E_{S|u}
\left[e^{\alpha(S,u)}\bar Z_\nu(S,u)\right]
\le1.
$$

再由 $\log$ 的凹性，

$$
\mathbb E_{S|u}\alpha(S,u)
\le
\mathbb E_{S|u}[-\log\bar Z_\nu(S,u)].
$$

对 $U$ 求期望并对 $\nu$ 取 infimum，得到弱对偶

$$
\sup_{\alpha\text{ 可行}}\mathbb E\alpha
\le F_S^{\mathrm{tilt}}(\lambda).
$$

而式 (23) 构造的 $\alpha_\lambda^*$ 可行，且其均值等于 $F_S^{\mathrm{tilt}}(\lambda)$。因此在最优解存在等条件下，

$$
\boxed{
F_S^{\mathrm{tilt}}(\lambda)
=
\sup_{\alpha\text{ 为 }S\text{-可测且满足式 (27)}}
\mathbb E\alpha(S,U).
}
\tag{30}
$$

这揭示了新路线与此前 Edge Dual 的准确关系：

> 条件化 D-tilted log-partition 不是另一个互不相关的界，而是“寻找最佳边端可测对偶函数”这一问题的等价原始表示。它的优势是自然给出 $\alpha^*$ 的构造，而不必先猜评分函数形状。

---

## 6. 与经典 D-tilted information 的关系

在率失真函数于 $D$ 可微、最优重建边缘 $P_{Q_D^*|U}$ 存在等正则条件下，令

$$
\lambda_D=-R_T'(D)\ge0.
$$

条件 D-tilted information 为

$$
\boxed{
\jmath_T(h,D|u)
=
-\log\mathbb E_{Q_D^*|u}
e^{-\lambda_D d(h,Q_D^*|u)}
-\lambda_DD.
}
\tag{31}
$$

在相应条件下

$$
\mathbb E[\jmath_T(H,D|U)]=R_T(D).
\tag{32}
$$

若 $S=H$ 且 $\nu_{\lambda_D}^*=P_{Q_D^*|U}$，则

$$
\alpha_{\lambda_D}^*(H,U)-\lambda_DD
=\jmath_T(H,D|U).
$$

若 $S$ 是更粗的边端统计量，则 $\alpha_\lambda^*(S,U)$ 是条件化后最佳的 $S$-可测对偶贡献，但一般不等于

$$
\mathbb E[\jmath_T(H,D|U)|S,U]
$$

或经典 D-tilted information 的简单回归。二者之间隔着 log-partition、条件期望和重新优化 $\nu$，不能把它们直接等同。

---

## 7. 固定码本大小的 one-shot converse

考虑给定 $U=u$ 后具有至多 $M$ 个消息的确定性单次编码器

$$
f_u:H\to\{1,\ldots,M\},
$$

以及解码后验

$$
q_m(u),\qquad m=1,\ldots,M.
$$

令实际重建为 $Q=q_{f_U(H)}(U)$。假设式 (27) 对几乎所有 $u$ 及所有 $q$ 成立。对任意 $\gamma\ge0$，定义

$$
\mathcal A
=
\left\{
\alpha_\lambda^*(S,U)-\lambda D
\ge\log M+\gamma
\right\}.
$$

在成功事件 $\{d(H,Q|U)\le D\}\cap\mathcal A$ 上，

$$
e^{\alpha_\lambda^*(S,U)-\lambda d(H,Q|U)}
\ge Me^\gamma.
$$

固定 $u$ 和消息 $m$，用 Markov 不等式及式 (27)：

$$
P\left[
f_u(H)=m,\ d(H,q_m|u)\le D,\ \mathcal A
\mid U=u
\right]
\le\frac{1}{Me^\gamma}.
$$

对至多 $M$ 个消息求和，再对 $U$ 平均：

$$
P[d(H,Q|U)\le D,\mathcal A]\le e^{-\gamma}.
$$

所以

$$
\boxed{
P[d(H,Q|U)>D]
\ge
P\left[
\alpha_\lambda^*(S,U)-\lambda D
\ge\log M+\gamma
\right]
-e^{-\gamma}.
}
\tag{33}
$$

式 (33) 是合法的 one-shot converse。它说明 $\alpha_\lambda^*-lambda D$ 的尾部分布会约束固定大小码本达到阈值失真的概率。

需要注意：

- 若码本大小依赖 $u$，应把 $\log M$ 改为 $\log M(u)$ 并逐条件处理；
- 随机编码器、公共随机性或可变长编码需要相应扩展，不能不加说明地直接套用式 (33)；
- 该结论仍不等于“每个样本必须使用 $\alpha/\ln2$ 个 bit”，因为 converse 是概率尾界而非逐样本确定性码长不等式。

---

## 8. 有限状态 $S$ 下的可计算形式

固定一个具有重复条件样本的完整导航条件 $u$，令

$$
S\in\{1,\ldots,J\},
\qquad
p_j(u)=P(S=j|U=u).
$$

记

$$
k_{\lambda,j}(q;u)
=
\mathbb E\left[
e^{-\lambda D_{\mathrm{KL}}(\Pi_F\|q)}
\mid S=j,U=u
\right].
\tag{34}
$$

则需解

$$
\boxed{
\min_{\nu\in\mathcal P(\Delta_m)}
-\sum_{j=1}^Jp_j(u)
\log\left[
\int k_{\lambda,j}(q;u)\nu(dq)
\right].
}
\tag{35}
$$

### 8.1 关于 $\nu$ 的凸性

对每个 $j$，映射

$$
\nu\mapsto\int k_{\lambda,j}(q;u)\nu(dq)
$$

是线性的，而 $-\log$ 为凸函数，所以式 (35) 是定义在概率测度凸集上的凸优化。

### 8.2 有限支持最优解

定义核向量

$$
v(q)
=
\bigl(k_{\lambda,1}(q),\ldots,k_{\lambda,J}(q)\bigr)
\in\mathbb R^J.
$$

目标仅通过

$$
z=\int v(q)\nu(dq)
$$

依赖 $\nu$，且 $z$ 位于 $\{v(q):q\in\Delta_m\}$ 的凸包中。Carathéodory 定理保证每个这样的 $z$ 可由至多 $J+1$ 个支持点表示。因此，只要最优 $z^*$ 存在，就存在支持点数不超过 $J+1$ 的最优 $\nu^*$。

这是一条存在性结论，并不自动给出这些支持点的位置；连续 $q$ 上仍需全局优化或有证书的列生成方法。

### 8.3 最优评分

求得 $\nu_\lambda^*$ 后，边端只需计算

$$
\boxed{
\alpha_{\lambda,j}^*(u)
=
-\log\int k_{\lambda,j}(q;u)\nu_\lambda^*(dq|u).
}
\tag{36}
$$

在线部署可把它实现为有限状态查表；完整教师只在离线核估计阶段运行。

---

## 9. 确定性动作的校验例

假设

$$
\Pi_F=\delta_B,
\qquad S=B,
$$

并取 $\lambda=1$。则

$$
e^{-D_{\mathrm{KL}}(\delta_b\|q)}=q(b),
$$

所以

$$
k_1(b;q)=q(b).
$$

设

$$
\nu^*=\sum_cP(B=c|u)\delta_{\delta_c}.
$$

则

$$
\int q(b)\nu^*(dq|u)=P(B=b|u),
$$

从而

$$
\boxed{
\alpha_1^*(b,u)
=-log P(B=b|U=u).
}
\tag{37}
$$

其均值为 $H(B|U)$。这与 log-loss 率失真函数

$$
R(D)=[H(B|U)-D]_+
$$

在有效区间内一致。该例验证了新公式能够自然退化为 Shannon 条件自信息量，而不是人为把 $-\log P(B|U)$ 塞进评分器。

但 AerialVLA 教师通常产生 soft posterior，且实际 $S$ 未必精确识别动作，因此式 (37) 是校验例，不应直接作为真实系统结论。

---

## 10. 辅助局部互信息率

由最优 $\nu_\lambda^*$ 定义辅助 Gibbs 信道

$$
P_\lambda^*(dq|s,u)
=
\frac{
\nu_\lambda^*(dq|u)k_\lambda(s,u;q)
}{
Z_\lambda^*(s,u)
}.
\tag{38}
$$

由式 (26) 在 $\nu_\lambda^*$ 支持上取等号，可验证该信道诱导的 $Q|U=u$ 边缘仍为 $\nu_\lambda^*(\cdot|u)$。

定义

$$
r_\lambda(s,u)
=D_{\mathrm{KL}}
\bigl(P_\lambda^*(Q|s,u)\|\nu_\lambda^*(Q|u)\bigr).
\tag{39}
$$

对 $\lambda>0$，使用式 (21) 和式 (38)：

$$
\begin{aligned}
r_\lambda(s,u)
&=
\mathbb E_{Q|s,u}
\left[
\log\frac{k_\lambda(s,u;Q)}{Z_\lambda^*(s,u)}
\right]\\
&=
\alpha_\lambda^*(s,u)
-\lambda
\mathbb E_{Q|s,u}
\widetilde d_\lambda(s,Q|u).
\end{aligned}
\tag{40}
$$

并且

$$
\boxed{
\mathbb E[r_\lambda(S,U)]
=I(S;Q_\lambda^*|U).
}
\tag{41}
$$

两种逐样本量的区别是：

| 量 | 严格含义 |
|---|---|
| $\alpha_\lambda^*(s,u)$ | 最优 $S$-可测率–失真 Lagrangian/对偶贡献 |
| $r_\lambda(s,u)$ | 辅助 Gibbs 信道的局部互信息贡献 |

$r_\lambda$ 是辅助条件化问题的局部率，不是原始视觉压缩系统的逐帧最小码长。将 $\alpha_\lambda^*$ 或 $r_\lambda$ 映射为实际 token 数 $K_t$，仍需预算校准或架构受限最优分配理论。

---

## 11. 数值近似为何必须再次认证

若只允许 $\nu$ 支持在有限候选集 $\{q_1,\ldots,q_L\}$ 上，式 (35) 的可行集合被缩小，因此算得的最小值一般**大于或等于**真正的 $F_S^{\mathrm{tilt}}$。这个偏大的数不能直接作为原问题下界。

对任意数值候选 $\widehat\nu_\lambda$，先定义原始分数

$$
\widehat\alpha_{\mathrm{raw}}(s,u)
=
-\log\widehat Z(s,u),
$$

其中

$$
\widehat Z(s,u)
=
\int k_\lambda(s,u;q)\widehat\nu_\lambda(dq|u).
$$

再定义真实总体违反因子

$$
C_\lambda(u)
=
\sup_q
\mathbb E\left[
e^{\widehat\alpha_{\mathrm{raw}}(S,u)-\lambda d(H,q|u)}
\mid U=u
\right].
\tag{42}
$$

若获得严格上界 $\overline C_\lambda(u)\ge C_\lambda(u)$，则

$$
\boxed{
\alpha_\lambda^{\mathrm{cert}}(s,u)
=
\widehat\alpha_{\mathrm{raw}}(s,u)
-\log\overline C_\lambda(u)
}
\tag{43}
$$

一定满足对偶约束，从而

$$
\boxed{
R_T(D)
\ge
\mathbb E[\alpha_\lambda^{\mathrm{cert}}(S,U)]
-\lambda D.
}
\tag{44}
$$

这里最稳妥的实践方式不是在同一数据上反复估计 $k$、选择 $\nu$、再报告 plug-in 结果，而是：

1. **训练集：** 估计核、选择 $S$、$\lambda$、候选支持点并求 $\widehat\nu$；
2. **校准集：** 冻结原始评分，直接用新样本认证式 (42) 的连续 $q$ supremum，并做式 (43) 的下移；
3. **测试集：** 对冻结的 $\alpha^{\mathrm{cert}}$ 计算 $\mathbb E\alpha$ 的下置信界。

若最终得到 $\underline\mu_\alpha$，则以相应联合置信度有

$$
\boxed{
R_T(D)
\ge
[\underline\mu_\alpha-\lambda D]_+.
}
\tag{45}
$$

### 连续 $q$ 的全局认证

只检查有限网格而不加覆盖修正是不够的。对 KL 核，

$$
e^{-\lambda D_{\mathrm{KL}}(\pi\|q)}
=e^{\lambda H(\pi)}\prod_aq_a^{\lambda\pi_a}.
$$

当 $0<\lambda\le1$ 时，它关于 $q$ 为凹函数，正权平均仍为凹函数，因此经验 separation 问题可以通过标准凸优化求全局最大值。总体统计认证还需要统一浓缩界、覆盖数或其他能同时控制所有 $q$ 的方法。

当 $\lambda>1$ 时，上述凹性一般不成立，需使用全局优化、分支定界或带严格误差项的概率单纯形覆盖。

---

## 12. 完整指令条件分布仍是不可回避的难点

所有对偶约束都要求对 $P_U$-几乎每个完整共享条件 $u$ 成立。若数据中每条完整指令只出现一次，那么经验条件分布会退化，无法凭有限样本得到非平凡、分布无关的逐条件总体证书。

不能未经证明把完整指令 $U$ 换成模糊方向类别 $V$。更粗的侧信息通常使率失真函数变大，对 $R(D|V)$ 的下界不一定是 $R(D|U)$ 的下界。

严谨方案是：

1. 固定完整目标描述和提示 $u$；
2. 在模拟器中使用不同地图、初始状态、随机种子生成独立 episode；
3. 从这些 episode 构造 $P(H|U=u)$ 的重复条件样本；
4. 以 episode 而非相邻帧为独立统计单位；
5. 明确证书首先适用于所定义的模拟器任务分布。

若要推广到真实飞行，还需真实样本上的重新认证或明确的分布迁移界。

---

## 13. AerialVLA 的推荐数值验证顺序

### 阶段 A：从低维任务开始

先把完整动作确定性粗化为 LAND/non-LAND、粗方向或三元任务变量。动作映射后的 KL 不大于完整动作 KL，因此粗任务率失真函数给完整任务提供合法但较松的下界。

### 阶段 B：构造条件数据

对多个固定完整指令生成独立 episode，保存

$$
(H_i,U_i,S_i,\Pi_{F,i},\text{episode id}).
$$

教师后验应固定 checkpoint、温度、动作序列联合概率的定义和后验平滑方式。

### 阶段 C：求解 log-partition 原始问题

对预先规定的 $\lambda\in(0,1]$：

1. 估计 $k_{\lambda,j}(q;u)$；
2. 用列生成或有限支持测度优化求候选 $\widehat\nu_\lambda$；
3. 计算 $\widehat\alpha_{\mathrm{raw}}$；
4. 在独立校准集完成全局对偶修正；
5. 在测试集报告式 (45)。

### 阶段 D：验证是否值得用于动态 $K$

分别比较 $\alpha_\lambda^{\mathrm{cert}}$、$r_\lambda$、简单动作熵、IB 信息增益和现有 Scorer，与下列量的关系：

- 各帧在不同 $K$ 下的教师动作 KL；
- LAND、急转、障碍规避等事件；
- 固定总 token 预算下的轨迹成功率、SPL 和真实通信 bit 数；
- seen、unseen object、unseen map 的分布外稳定性。

只有这些实验显示评分确有区分度后，才有充分理由把它用于

$$
K_t=g(\alpha_t)
\quad\text{或}\quad
K_t=g(r_t).
$$

映射 $g$ 仍是预算控制器，不是当前率失真证明自动给出的逐帧最优编码定理。

---

## 14. 最终可正式写入论文的结论

可以严谨宣称：

1. AerialVLA 教师动作后验 KL 是额外期望 log-loss；
2. 条件任务率失真函数给任意满足任务失真约束的实际消息互信息和平均可唯一解码码长提供下界；
3. 通过对 Gibbs log-partition 关于边端统计量 $S$ 做条件 Jensen，可得到严格下界 $F_S^{\mathrm{tilt}}(\lambda)$；
4. 该优化的最优解自然诱导最优的 $S$-可测 Csiszár 对偶函数 $\alpha_\lambda^*$；
5. $\alpha_\lambda^*$ 同时进入平均率下界和固定码本大小的 one-shot converse；
6. 有限状态 $S$ 下存在至多 $J+1$ 个支持点的最优重建边缘分布；
7. 任意数值近似都可通过违反因子 $\overline C_\lambda$ 下移为合法 witness。

尚不能宣称：

1. $\alpha_t/\ln2$ 是第 $t$ 帧确定需要发送的最少 bit；
2. $r_t$ 是原始 AerialVLA 视觉压缩器的真实逐帧最小码率；
3. $K_t\propto\alpha_t$ 或 $K_t\propto r_t$ 已由理论证明最优；
4. 在有限候选 $q$、训练样本或模拟器上得到的 plug-in 数值无需校准即可推广到真实任务分布；
5. 模型无关信息论下界能够直接证明某个具体的 $K=64$ 或 $K=128$ 最优。

最准确的主线是

$$
\boxed{
\text{教师 KL 任务}
\to F_H(\lambda)
\to \text{Gibbs log-partition}
\to \text{条件 Jensen}
\to F_S^{\mathrm{tilt}}(\lambda)
\leftrightarrow \alpha_\lambda^*
\to \text{统计认证}
\to [\underline\mu_\alpha-\lambda D]_+.
}
$$

与此前方案相比，新的实质进步不是“摆脱了对偶认证”，而是找到了产生最佳边端可测对偶 witness 的原则性方法，并把平均下界、D-tilted information、one-shot converse 和可计算有限状态优化统一在同一数学结构中。

---

## 参考文献

1. Kostina, V., & Verdú, S. *Fixed-Length Lossy Compression in the Finite Blocklength Regime*. IEEE Transactions on Information Theory, 58(6), 3309–3338, 2012. https://arxiv.org/abs/1102.3944
2. Kostina, V., & Verdú, S. *Nonasymptotic Noisy Lossy Source Coding*. IEEE Transactions on Information Theory, 62(11), 6111–6123, 2016. https://arxiv.org/abs/1401.5125
3. Le, S.-Q., Tan, V. Y. F., & Motani, M. *Second-Order Coding Rates for Conditional Rate-Distortion*. IEEE Transactions on Information Theory, 61(4), 2015. https://arxiv.org/abs/1410.2687
4. Yang, Y., & Mandt, S. *Towards Empirical Sandwich Bounds on the Rate-Distortion Function*. ICLR 2022. https://arxiv.org/abs/2111.12166
5. Tishby, N., Pereira, F. C., & Bialek, W. *The Information Bottleneck Method*. https://arxiv.org/abs/physics/0004057
6. Courtade, T. A., & Weissman, T. *Multiterminal Source Coding under Logarithmic Loss*. IEEE Transactions on Information Theory, 60(1), 740–761, 2014. https://arxiv.org/abs/1110.3069
7. Xu, P., Deng, Z., Deng, J., Gu, Z., & Wan, S. *AerialVLA: A Vision-Language-Action Model for UAV Navigation via Minimalist End-to-End Control*. https://arxiv.org/abs/2603.14363
