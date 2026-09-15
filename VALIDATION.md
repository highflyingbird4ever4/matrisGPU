# Validation — 2026-09-14

## 本地契约测试

`python -m pytest -q`：**8 passed, 1 skipped**。跳过的是需显式开启的真实GPU测试；真实GPU对照随后单独执行通过。

通过项包括带/不带晶胞弛豫的原生ASE FIRE逐轨迹对照，候选级构图失败隔离，输入不修改与序号保持，能量/应力单位，惰性输入与补位，零步/步数上限与末态轨迹模式，以及无关异常向外传播。

本地Torch为2.11.0+cu126。ASE提示未来3.28将移除`step(forces)`参数；本发行版依赖上限明确为`ase<3.28`，保持已验证路径。

## 真实A100短对照

在已冻结MatRIS环境中运行 `python tests/test_real_gpu.py`：**PASS**。
上游commit为`c16f569ca08e6905e91b64e2ee68614303e46f7f`，Torch2.6.0+cu124，模型matris_10m_oam，FP32/TF32关闭。

- 两个结构：扰动Cu和Fe晶胞，batch_size=2。
- E/F/S/m对照上游MatRISCalculator：总能量绝对容差1e-4 eV，力分量2.5e-4 eV/Å，应力分量1e-5 eV/Å³，磁矩1e-4。
- 两个独立FIRE、FrechetCellFilter轨迹最多3步，对照上游README的StructOptimizer；终态原子坐标及晶胞绝对容差1e-4 Å。
- 实际测试的core.py SHA256：`408c33142627565b24f15194500cb965f4dd2c3d6ca3095fb08afb066e6ea522`。

测试调用的是仓库源码，未安装/修改远端计算环境。首次pytest调用因远端未安装pytest而未执行测试；随后直接运行同一真实GPU检查函数，通过。

## 打包与边界

使用`python -m pip wheel . --no-deps --no-build-isolation -w dist`构建wheel。
新包提取了已验证的独立FIRE批量核心，但新增了流式输入、通用接口与真实收敛status；没有重新执行1134结构全矩阵，也没有把旧实验的吞吐数字当作新包性能验收值。
没有运行生产切片、数据库写入、DFT或发布远端仓库。
