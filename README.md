# matrisGPU

一个共享 MatRIS 模型，批量预测多个结构，并维护各自独立的 FIRE 弛豫状态。
接口参考 [MatRIS README](https://github.com/HPC-AI-Team/MatRIS/blob/c16f569ca08e6905e91b64e2ee68614303e46f7f/README.md)，不是完整替代 MatRIS。
不包含模型权重、数据库连接、集群调度或研究项目的数据。

## 安装

先安装适合本机 CUDA 的 PyTorch，然后安装已验证版本的上游 MatRIS：

```bash
python -m pip install "git+https://github.com/HPC-AI-Team/MatRIS.git@c16f569ca08e6905e91b64e2ee68614303e46f7f"
python -m pip install -e .
```

Python导入名为`matrisgpu`。模型加载和权重缓存由MatRIS负责。已在已有MatRIS环境中运行时，无需重新安装上游。

## 批量单点预测

```python
import torch
from ase.build import bulk
from matrisgpu import BatchCalculator

torch.set_num_threads(4)
calc = BatchCalculator(
    model="matris_10m_oam",
    task="efsm",
    device="cuda",
    batch_size=32,
)
atoms = [bulk("Cu", a=3.7, cubic=True), bulk("Fe", a=2.9, cubic=True)]
results = calc.predict(atoms)
for result in results:
    if result["status"] == "failed":
        print(result["index"], result["error"])
        continue
    energy = result["energy"]       # 总能量，eV；不是每原子能量
    forces = result["forces"]       # (N, 3)，eV/Å
    stress = result["stress"]       # eV/Å³，通常为(3, 3)，上游Voigt输出则保留(6,)
    magmoms = result["magmoms"]     # (N,)，模型磁矩
```

`predict(单个Atoms或Structure)`返回一个字典；传入可迭代对象则返回输入顺序的字典列表。
`ipredict(...)`逐条返回结果，最多读取一个batch的输入。`BatchCalculator`本身不是ASE单结构Calculator，不能赋给`atoms.calc`；原生ASE用法仍使用上游`MatRISCalculator`。

## 结构优化：参考 MatRIS README 的调用方式

```python
from matrisgpu import StructOptimizer

matris_opt = StructOptimizer(
    model="matris_10m_oam",
    task="efsm",
    optimizer="FIRE",
    device="cuda",
    batch_size=32,
)
opt_results = matris_opt.relax(
    atoms=atoms,                   # 列表；单个ase.Atoms/pymatgen.Structure也支持
    verbose=False,
    steps=500,
    fmax=0.01,                     # 默认0.05，与上游示例保持一致；本项目测试用0.01
    relax_cell=True,
    ase_filter="FrechetCellFilter",
    mask=[1, 1, 0, 0, 0, 1],     # 二维实验用；默认None，即不施加此mask
)
for opt_result in opt_results:
    if opt_result["status"] != "success":
        print(opt_result["index"], opt_result["status"])
        continue
    trajectory = opt_result["trajectory"]
    energy = trajectory.energies[-1]
    force = trajectory.forces[-1]
    stress = trajectory.stresses[-1]  # ASE Voigt顺序，eV/Å³
    magmom = trajectory.magmoms[-1]
    final_structure = opt_result["final_structure"]
```

单结构调用返回字典，与上游示例相同；列表调用返回输入顺序的列表。
每个结构独立判断收敛并退出，新结构随后补入空位，不必等同批所有结构都收敛。
不修改调用者的输入结构；默认在输出结构添加预测magmom，保留其他位点属性。

## 一万个结构的切片

使用`irelax`，避免保留整片轨迹。它按**完成顺序**返回，`index`始终是输入序号。
`keep_trajectory=False`只保留每个结构最后一个已计算状态；默认True记录每个已计算状态，不重复追加终态。

```python
import json
from pathlib import Path
from monty.json import MontyEncoder
from pymatgen.core import Structure

paths = sorted(Path("inputs").glob("*.cif"))
structures = (Structure.from_file(path) for path in paths)
with open("results.jsonl", "x", encoding="utf-8") as output:
    for result in matris_opt.irelax(
        structures, steps=500, fmax=0.01, relax_cell=True,
        mask=[1, 1, 0, 0, 0, 1], keep_trajectory=False,
    ):
        record = {
            "input": str(paths[result["index"]]),
            "index": result["index"],
            "status": result["status"],
            "steps": result["steps"],
            "converged": result["converged"],
            "error": result.get("error"),
            "final_structure": result["final_structure"].as_dict(),
        }
        output.write(json.dumps(record, cls=MontyEncoder) + "\n")
        output.flush()
```
