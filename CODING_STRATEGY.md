# Molecular Docking 自动化分析软件 — 最优Coding策略

> **生成时间**: 2026-05-22
> **目标**: 顶刊水平（Nature Methods / J. Med. Chem. / JCIM）自动化分子对接分析
> **开发模式**: Kimi Code CLI（非OpenClaw）

---

## 一、三方调研结论

### 1.1 Vibe Coding策略文档（Molecular_Docking_Environment_Report.md）

**文档质量**: ⭐⭐⭐⭐⭐ 极其详尽，1831行，覆盖从环境配置到发表级全流程

**核心成果**:
- 9阶段顶刊Pipeline架构（受体准备→配体准备→口袋探测→对接→验证→相互作用→MD→自由能→可视化）
- 完整的配置模板（config.yaml）
- 详细的顶刊注意事项（redocking验证、共识打分、exhaustiveness≥32等）
- 自动化开发路线图（MVP→V3.0）
- 丰富的代码片段和命令参考

**可直接采纳**:
- Pipeline 9阶段架构
- 配置模板设计
- 发表级参数标准（exhaustiveness=32, n_poses=20等）
- Redocking验证流程
- 图表生成规范

### 1.2 Conda `autodock` 环境核实

| 工具 | 状态 | 版本/路径 | 备注 |
|------|------|----------|------|
| Python | ✅ | 3.12 | |
| RDKit | ✅ | 2026.03.1 | pip |
| Vina (Python API) | ✅ | 1.2.7 | conda-forge |
| Vina (CLI) | ✅ | vina, vina_split | conda bin |
| Meeko | ✅ | 0.7.1 | pip |
| PLIP | ✅ | 3.0.0 | conda-forge |
| OpenBabel (Python+CLI) | ✅ | 3.1.0 | pip wheel |
| MDAnalysis | ✅ | 2.10.0 | conda-forge |
| ProLIF | ✅ | 2.1.0 | pip |
| PoseBusters | ✅ | 0.6.5 | pip |
| fpocket (CLI) | ✅ | 4.2.2 | conda bin |
| P2Rank (CLI) | ✅ | 2.5.1 | conda opt/ + bin symlink |
| PyMOL (CLI) | ✅ | conda bin | `pymol`可用 |
| PyMOL (Python import) | ⚠️ | conda env | `import pymol`在conda中需特殊处理 |
| GROMACS (CLI) | ✅ | 2026.0 | conda bin |
| gmx-MMPBSA | ✅ | 1.6.4 | pip |
| OpenMM | ✅ | 8.5.1 | conda-forge |
| Biopython | ✅ | 1.87 | `from Bio import PDB` |
| pandas/numpy/scipy | ✅ | latest | |
| matplotlib/seaborn | ✅ | latest | |
| reportlab/cairosvg | ✅ | latest | |
| pubchempy/ChEMBL/pypdb | ✅ | latest | |
| Java | ⚠️ | 未找到系统Java | 但conda无Java，P2Rank需Java Runtime |
| LigPlot+ | ✅ | ~/ligplus/ | hbadd/hbplus/ligplot可用 |

**关键发现**: PyMOL的Python import在conda环境中不稳定（`import pymol`失败），但`pymol` CLI命令可用。策略：3D渲染采用**CLI调用模式**（subprocess调用pymol -cq -d），而非Python API嵌入。

### 1.3 GitHub仓库（rongtianhua/Autodock-Skill）分析

**仓库规模**: 30+ Python模块，~8000行代码，测试文件15+

**设计亮点（值得借鉴）**:
1. `DockingResult` dataclass — 结构化结果容器，支持JSON/CSV序列化
2. 共识打分 — Vina + AD4 + Vinardo三函数中位数
3. 多口袋自动探测 — fpocket + P2Rank + 排序验证
4. 多构象对接 — 10个独立构象分别对接取最优
5. PLIP单一权威 — 不再fallback到RDKit几何检测
6. PyMOL Leipzig标准 — 发表级虚线参数
7. LigPlot+ CLI集成 — 无需GUI
8. SnakeMake工作流 — 大规模虚拟筛选

**代码缺陷（必须避免）**:
1. **环境硬编码** — 到处写死`autodock313`路径，而用户环境是`autodock`
2. **工具路径硬编码** — fpocket、P2Rank、Java路径都是绝对路径
3. **过度复杂** — 30+模块互相纠缠，单个模块2000+行
4. **OpenClaw耦合** — `~/.openclaw`路径依赖、PYTHONPATH hacks
5. **错误处理粗糙** — 大量`except Exception: pass`，bug被静默吞掉
6. **测试不足** — 虽有tests但覆盖率不够，很多功能未验证
7. **PyMOL import失败** — 未处理conda环境下pymol不可用的情况
8. **文档与代码脱节** — SKILL.md描述的API和实际代码不完全一致

---

## 二、最优Coding策略

### 2.1 核心原则

| 原则 | 说明 |
|------|------|
| **稳健优先** | 每个外部调用都必须有try/except + 日志 + 降级策略，绝不静默吞异常 |
| **环境自适应** | 通过`shutil.which()`或`conda env`变量自动发现工具，零硬编码路径 |
| **科学准确** | 严格遵循文档中的顶刊标准（exhaustiveness≥32, redocking验证, 共识打分） |
| **模块化低耦合** | 每个模块可独立导入和测试，模块间通过明确接口通信 |
| **测试驱动** | 每写一个函数就写一个测试，端到端pipeline用真实数据验证 |
| **MVP先行** | 先跑通核心pipeline（准备→口袋→对接→验证→相互作用→渲染），再扩展MD/MMPBSA |

### 2.2 项目架构

```
autodock/
├── __init__.py          # 公开API导出
├── __main__.py          # CLI入口 (python -m autodock)
├── core.py              # 异常体系、日志、DockingResult、常量、环境探测
├── config.py            # YAML配置解析与验证
├── utils.py             # 通用工具（路径、文件、坐标计算等）
├── preparation.py       # 受体准备(PDBFixer/Meeko)、配体准备(RDKit+Meeko)、口袋探测(fpocket+P2Rank)
├── docking.py           # Vina对接、批量对接、共识打分
├── validation.py        # PoseBusters、RMSD、clash检测、redocking验证
├── interactions.py      # PLIP/ProLIF相互作用检测
├── rendering.py         # PyMOL 3D (CLI调用) + RDKit 2D (Cairo)
├── reporting.py         # PDF/Excel报告生成
└── tests/               # pytest测试集
    ├── test_core.py
    ├── test_preparation.py
    ├── test_docking.py
    ├── test_validation.py
    ├── test_interactions.py
    └── test_end_to_end.py
```

### 2.3 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| PyMOL调用模式 | **CLI subprocess** (`pymol -cq -d`) | conda环境中`import pymol`不稳定，CLI始终可用 |
| 口袋探测 | **fpocket + P2Rank** | fpocket几何+ P2Rank ML，互补验证 |
| 相互作用检测 | **PLIP单一权威** | 不再fallback到RDKit几何检测（github教训） |
| 2D渲染 | **RDKit Cairo + PLIP数据** | 不依赖LigPlot+（可选增强） |
| 受体准备 | **Meeko Polymer + PDBFixer** | 比Open Babel更精确，支持非标准残基 |
| 配体准备 | **RDKit ETKDGv3 + Meeko** | 顶刊标准，可重复seed |
| 共识打分 | **Vina + AD4 + Vinardo 中位数** | 降低单一打分函数偏差 |
| 报告格式 | **PDF (reportlab) + Excel (openpyxl)** | 发表级矢量图 + 数据表格 |

### 2.4 环境自适应机制

```python
# core.py 中的环境探测 — 零硬编码
import shutil
import os
import subprocess

def find_conda_tool(name: str) -> str | None:
    """在conda环境中查找工具，优先当前conda env的bin目录。"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidate = os.path.join(conda_prefix, 'bin', name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which(name)

def find_java() -> str | None:
    """查找Java运行时，优先conda env中的openjdk。"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        for jdk_ver in ['openjdk', 'openjdk@21', 'openjdk@25']:
            candidate = os.path.join(conda_prefix, 'bin', 'java')
            if os.path.exists(candidate):
                return candidate
    # fallback: /usr/libexec/java_home
    try:
        result = subprocess.run(['/usr/libexec/java_home', '--failfast'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            java_home = result.stdout.strip()
            candidate = os.path.join(java_home, 'bin', 'java')
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    return shutil.which('java')

# P2Rank路径探测
_P2RANK_CANDIDATES = [
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'opt', 'p2rank_2.5.1', 'prank'),
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin', 'prank'),
    shutil.which('prank'),
]
```

### 2.5 防御性编程模式

```python
# 每个外部调用都必须遵循此模式
def safe_external_call(cmd: list, timeout: int = 120, 
                        fallback_result=None, 
                        error_msg: str = "External call failed") -> tuple:
    """
    安全调用外部命令。
    
    Returns:
        (success: bool, result, error: str|None)
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                                timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout, None
        else:
            logger.warning(f"{error_msg}: {result.stderr[:200]}")
            return False, fallback_result, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"{error_msg}: timeout after {timeout}s")
        return False, fallback_result, f"timeout"
    except FileNotFoundError:
        logger.error(f"{error_msg}: command not found ({cmd[0]})")
        return False, fallback_result, f"command not found"
    except Exception as e:
        logger.error(f"{error_msg}: unexpected error: {e}")
        return False, fallback_result, str(e)
```

### 2.6 发表级结果交付体系

| 交付物 | 标准 | 工具 |
|--------|------|------|
| 3D结合模式图 | 300dpi PNG, 2400×2400 | PyMOL CLI ray tracing |
| 2D相互作用图 | 300dpi PNG/PDF矢量 | RDKit Cairo + PLIP数据 |
| 对接结果表 | CSV + Excel | pandas + openpyxl |
| 相互作用指纹 | 热力图 PDF | seaborn |
| 综合报告 | PDF (A4) | reportlab |
| Pose验证 | PoseBusters通过报告 | posebusters |
| Redocking验证 | RMSD<2.0Å | spyrmsd |

### 2.7 开发顺序与里程碑

| 阶段 | 模块 | 关键功能 | 验证标准 |
|------|------|---------|---------|
| **Phase 0** | 策略文档 | 完成本文件 | — |
| **Phase 1** | core + config + utils | 异常体系、日志、配置解析、环境探测 | `python -c "import autodock; autodock.status()"` |
| **Phase 2** | preparation | 受体准备、配体准备、口袋探测 | 6LU7受体→PDBQT, N3配体→PDBQT, 口袋中心与文献一致 |
| **Phase 3** | docking | Vina对接、批量、共识打分 | Redocking RMSD<2.0Å |
| **Phase 4** | validation | PoseBusters、RMSD、clash | 100% PoseBusters通过 |
| **Phase 5** | interactions | PLIP检测 | 检出H-bond+疏水+π-π |
| **Phase 6** | rendering | 3D PyMOL + 2D RDKit | 300dpi PNG生成 |
| **Phase 7** | reporting | PDF + Excel | 完整报告生成 |
| **Phase 8** | CLI + end-to-end | 一键运行 | `python -m autodock run --receptor 6LU7 --ligand N3` |

---

## 三、与GitHub仓库的关系

| 方面 | 策略 |
|------|------|
| **代码复用** | **不复用**。只借鉴宏观架构思路（DockingResult设计、共识打分、多口袋策略）。全部重写，确保零bug继承。 |
| **API设计** | 参考github的公开API（`dock_ligand()`, `prepare_receptor()`等），但简化参数，增强类型注解。 |
| **配置格式** | 完全采用文档中的YAML模板，不沿用github的`docking_config.template.yml`。 |
| **测试策略** | github有tests但覆盖不够。我们采用**真实数据端到端测试**（6LU7+N3），每个模块独立测试。 |

---

## 四、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| PyMOL CLI渲染失败 | 3D图无法生成 | 降级为RDKit 3D (NGLView)或纯2D图；提前检测pymol可用性 |
| P2Rank需要Java | 口袋探测降级 | 环境探测自动找Java；找不到则只用fpocket |
| PLIP解析失败 | 相互作用检测失败 | 降级到ProLIF；两者都失败则报告并继续 |
| Vina对接超时 | 大体系或高exhaustiveness | 默认timeout=600s；可配置；超时自动降级exhaustiveness重试 |
| 非标准残基 | 受体准备失败 | Meeko `allow_bad_res=True`自动移除；日志告警 |

---

**策略制定完成，立即进入Phase 1构建。**
