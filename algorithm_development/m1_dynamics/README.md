# M1：具有明确未来映射的角–时延–Doppler模型

开发开始于2026-10-29；2026-10-30加入同条件复数SBL。当前内容是单径／双径可控模型开发，不是96路线数据集的正式未来预测评测。

## 1. 要解决的问题

根据过去4 ms的已知OFDM导频及接收波形，预测截止时刻之后2 ms的传播CSI。与M0不同，本模型让**同一组参数同时决定历史接收波形和未来信道**。

设参考时刻为 t_ref；在所考察的短窗口内，每条路径具有固定的复增益g_l、时延τ_l、出发/到达方向以及传播Doppler ν_l。复增益是参考阵元在t_ref的基带系数，已包含载波传播相位。给定数组响应a_tx、a_rx，传播信道为

\[
H_{r,q}(f,t)=\sum_{l=1}^{L}g_l\,a^{\rm rx}_{r,l}a^{\rm tx}_{q,l}
e^{j2\pi\nu_l(t-t_{\rm ref})}e^{-j2\pi f\tau_l}.
\]

这里ν以Hz为单位，f是相对载频的基带频率。当前采用与原数据一致的发射端转置约定：H乘波束，不再对波束额外共轭。

这是常参数局部相位模型。恒定时延与非零Doppler并存，是短窗内冻结包络时延、保留快速载波相位变化的近似；不声称它精确描述任意6 ms真实运动、遮挡或路径生灭。人工样例不借用原数据的局部准入证书。

## 2. 历史接收算子与未来映射

令x_q(t)是第q根发射天线的已知基带波形。每个CP-OFDM符号在其含CP的实际时间区间内是有限傅里叶多项式，区间外为零。接收样本为

\[
y_r(t_n)=e^{j2\pi\epsilon t_n}
\sum_l g_l a^{\rm rx}_{r,l}e^{j2\pi\nu_l(t_n-t_{\rm ref})}
\sum_q a^{\rm tx}_{q,l}x_q(t_n-\tau_l)+n_r(t_n).
\]

ε是公共振荡器CFO；ν_l是传播Doppler。公共CFO采用绝对时间相位e^(j2πεt)，与已有接收机一致。噪声方差表示每个复样本的E|n|²，单位FFT不会改变它。

对候选角度、时延、Doppler字典逐列计算这个波形，再FFT并展平，可得

\[
\boxed{\mathbf y=\boldsymbol\Phi_{\mathcal P}(\epsilon)\mathbf g+\mathbf n,
\qquad \mathbf h_* =\mathbf B_*(t_*,\mathcal F_*)\mathbf g.}
\]

- Φ的行顺序为[symbol,rx,FFT bin]，维度(S·Nr·N)×G。
- B*的行顺序为[rx,tx,target frequency]，维度(Nr·Nt·F*)×G。
- P包括导频符号、波束、发射时刻、采样率与CP。
- B*直接在目标频率上求值，因此支持不在FFT网格上的目标；不使用最近bin代替。
- 给定参数字典及CFO，对g线性；同时估计字典参数/CFO时仍是非线性逆问题。

实现 [path_model.py](path_model.py) 的 `observation_basis` 保留分数时延、真实符号支持及逐接收样本相位，因此保留ICI和CP不足时的ISI。`future_basis`定义输出目标，`predict`检查目标所需的频率参考是否存在。

## 3. 未知CFO的可辨识边界

这里的结论只针对“每径Doppler自由未知、复增益自由未知、没有外部绝对频率约束”的上述模型。对于任意实数δ，作变换

\[
\epsilon'=\epsilon+\delta,\qquad
\nu_l'=\nu_l-\delta,\qquad
g_l'=g_l e^{-j2\pi\delta t_{\rm ref}}.
\]

代回接收式，所有时刻的无噪声接收波形完全相同；FFT、增加导频或改变波束也不能改变这一代数等价性。但是

\[
H'(f,t_*)=e^{-j2\pi\delta t_*}H(f,t_*).
\]

只要真实H非零，相对平方误差为4 sin²(πδt*)。本次δ=125 Hz、t*=6 ms，实测等于2，与理论一致；观测相对误差约8.00×10⁻³²。含公共CFO的有效信道e^(j2πεt*)H(f,t*)保持相同。

这说明**当前自由参数模型不能仅凭导频唯一分离公共CFO与所有传播Doppler**。这是频率参考对称性的直接结果，不是新颖性声明，也不是对加入几何/运动硬约束后模型的不可辨识性证明。有限CFO先验可以产生贝叶斯推断，但它没有让似然变得唯一。

若t_ref=0，在实参数Jacobian中，CFO方向的列等于所有逐径Doppler方向列之和，因此还有局部秩亏；有噪声时增加优化迭代不能消除这个结构性方向。

当前API明确区分：

| 模式 | 可使用的信息 | 输出含义 |
|---|---|---|
| `physical_known_cfo` | 外部给定CFO，逐样本补偿 | 传播Doppler及传播CSI；本轮为已知CFO诊断 |
| `total_received` | CFO未知 | 总频率ρ_l=ν_l+ε及有效CSI |
| `total_received` + 显式CFO锚 | 外部校准或已关联参考径的已知Doppler | 条件性的传播CSI |

未知CFO却请求传播CSI、又未提供锚时，`predict`会抛出错误；OMP与SBL都遵守此规则。本轮“已知参考径Doppler”示例是oracle诊断，不是已经从实际地图估计出的CFO。

这里的绝对相位以全局t=0约定为基础，误差式含t*，不能擅自改成仅含2 ms预测时距。若引入额外的截止时刻传播相位校准，应重新定义条件问题。对公共相位不敏感的波束指标可能不受这个等价类影响，不能把传播CSI复数NMSE的歧义直接解释为波束性能损失。

## 4. 不要把两种频率歧义混为一谈

简化标量采样y_n=a exp(j2πνnΔ)在Δ=1 ms时，ν和ν+1000 Hz相同。其未来如果仍在1 ms网格上也相同；本次6 ms目标的别名误差约2.16×10⁻³⁰，而4.5 ms目标的误差等于4。

但实际协议保留整个OFDM符号内的接收样本。对本次完整三burst波形，单独把Doppler增加1000 Hz、保持CFO不变，观测NMSE约0.347，而非0。因此不能把1 ms单点采样结论直接套在全FFT观测上。上一节公共CFO与自由Doppler的联合平移则在所有接收样本上成立。

## 5. 当前两种稀疏估计器

OMP对同一固定的405列波形字典做归一化相关性选择，再联合LS。最多3个原子是预先声明的预算，单径、双径与离网样例一致；不读取真实路径数。噪声能量用于停止。

SBL实现于 [sbl.py](sbl.py)。先将Φ列归一化为Ψ，并用u吸收列范数。设

\[
\mathbf y=\boldsymbol\Psi\mathbf u+\mathbf n,\quad
\mathbf u\mid\boldsymbol\gamma\sim\mathcal{CN}(0,\boldsymbol\Gamma),\quad
\boldsymbol\Gamma=\operatorname{diag}(\boldsymbol\gamma).
\]

已知噪声方差σ²时，复数高斯后验为

\[
\boldsymbol\Sigma=(\boldsymbol\Gamma^{-1}+\boldsymbol\Psi^H\boldsymbol\Psi/\sigma^2)^{-1},
\qquad \boldsymbol\mu=\boldsymbol\Sigma\boldsymbol\Psi^H\mathbf y/\sigma^2,
\qquad \gamma_j^{new}=|\mu_j|^2+\Sigma_{jj}.
\]

最后恢复物理系数g，再用同一个B*预测。代码通过Cholesky和405维原子空间统计量计算，不建立7296×7296接收协方差；没有硬支撑裁剪。最多300轮，均值与负对数证据相对变化均小于10⁻⁵时停止，达到上限会明确记录未满足停止准则。

这里γ是**方差**；Bu原文相应γ是**精度**，不能照抄符号后混用更新公式。所报告后验协方差仅条件于当前估计的超参数、字典、给定噪声与CFO，不含频率或超参数不确定性，不能直接当作校准的未来风险。

这是一组通用复数Gaussian-EM SBL基线，**不等于Bu WCM算法复现**。后者还涉及WCM分层先验、频偏/时延偏移/噪声参数的联合更新及其特定观测模型。参见 [Wipf–Rao基础论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/01315936.pdf)；Bu对应部分是原文Sec.IV、式(47)–(55)及后续参数更新。

## 6. 已完成的验证与范围

run_01/results.json（本地运行记录，未随仓库发布） 保存了M1第一轮控制：

- 正常CP及短CP条件下，分别对照原接收机和独立傅里叶多项式直接求值；波形NMSE约10⁻³¹与10⁻²⁸。
- 非零参考时刻下的未来映射、原生信道重构及已知支撑LS一致。
- 额外核对57符号×16接收天线×256 FFT的原尺寸前向模型，NMSE约9.10×10⁻³¹。
- 已知CFO的单径/双径在网格内、无噪声时，未来NMSE约4.2×10⁻³¹。
- 离网单径失败被保留：OMP无噪声未来NMSE约2.159，而非人为修改网格让它通过。
- 未知CFO的连续等价类、外部参考径诊断及粗采样别名的适用范围分别测试。

推断比较使用缩小的4Tx/2Rx、64 FFT、7.68 MHz采样率；保持120 kHz子载波间隔、2.083 μs CP及3×19符号/4 ms时间结构。目标为6 ms时刻的16个频率点。原尺寸前向核对不代表原尺寸全字典搜索已经完成。

噪声实验为三个固定样例、0/10/20 dB、每个5个噪声种子。这里SNR按人工样例接收功率定义，与原devset的历史功率定标不同。匹配网格、已知CFO和7296个复观测使正控制非常容易；不能以其很低的NMSE宣传实际性能。

SBL与OMP的配对实验由 [run_sbl_comparison.py](run_sbl_comparison.py) 运行。它逐项核对OMP结果与原45个控制样例一致，同时保存SBL的证据轨迹、收敛标记和运行时间。`completed_trials.jsonl`只表示已完成的部分；最终`results.json`中的status=passed才确认整组完成。

2026-10-30的 `run_03_sbl_20261030/results.json` 已完整通过，共45对、90行，耗时328.24秒。SBL中34/45次满足所声明停止准则，11次达到300轮上限。20 dB离网单径的平均NMSE为OMP +3.3444 dB、SBL +3.6041 dB；当前固定网格SBL没有解决该反例。未把此控制解释为OMP普遍优于SBL，也未把它称为Bu算法复现。完整数值与局限见 开发结果与边界（本地运行记录，未随仓库发布）。

新增 [grid_tools.py](grid_tools.py) 只实现Bu式两段非均匀网格、左端点加区间内比例的时延编码及中心化DFT变换，尚未接入连续参数联合估计。`run_grid_checks.py` 已验证区间边界、完整DFT离格表示、单时延单频率的表示反例以及现有周期地图的时延歧义。详见 Bu网格适配说明（本地运行记录，未随仓库发布）。

## 7. 运行与下一步

在工作区根目录运行，输出目录须为尚不存在的新目录：

```powershell
& '.\sionna_small_validation\.venv\Scripts\python.exe' `
  '.\algorithm_development\m1_dynamics\run_controls.py' `
  --out '.\output\algorithm_m1_20261029\my_controls'

& '.\sionna_small_validation\.venv\Scripts\python.exe' `
  '.\algorithm_development\m1_dynamics\run_sbl_comparison.py' `
  --out '.\output\algorithm_m1_20261029\my_sbl_comparison'

& '.\sionna_small_validation\.venv\Scripts\python.exe' `
  '.\algorithm_development\m1_dynamics\run_grid_checks.py' `
  --out '.\output\algorithm_m1_20261029\my_grid_checks'
```

依赖使用现有NumPy、SciPy、h5py环境；未运行新RT，未读test split，也未修改原release或M0。

下一步优先处理连续时延/角度/Doppler细化，以及来自可用地图与带噪运动报告的频率约束。先分别测量“固定网格→连续细化”“无地图→地图约束”的增益，再考虑生成先验。Bu式分层先验和联合估计需单独适配、核验，不混入这组最基础的SBL比较。
