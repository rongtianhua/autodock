# autodock 分子对接管道 — 综合审计报告

**审计日期**: 2026-06-08  
**审计范围**: 26 个模块，~21,000 LOC  
**审计维度**: 算法正确性 · 数值稳定性 · 物理单位 · 化学正确性 · 可复现性 · 边界情况 · 文件格式鲁棒性 · 代码质量 · 测试覆盖 · 文档完整性 · 顶刊发表标准符合度  
**基准状态**: 20 靶点多样性基准，top-1 = 55%，best-achievable = 85%，9 个持续失败靶点

---

## 执行摘要

`autodock` 是一个工程精良、面向发表的分子对接框架，具备广泛的容错链、`spawn` 多进程安全、超时架构和文献支持的科学常数。自 2026-05-30 审计以来，**3 项关键问题已修复**（Kabsch RMSD、MM-GBSA NaN 级联、CI 矩阵扩展），但**10 项问题仍然开放**，其中包括 6 个评分失败靶点和 3 个 MM-GBSA 模板不匹配。

**总体评级**: 代码质量 **B+**（工程），科学严谨性 **B**（算法正确），发表卫生 **C+**（文档与元数据）。

| 严重性 | 数量 | 主题 |
|--------|------|------|
| 🔴 **严重** | 3 | 评分失败、MM-GBSA 模板不匹配、tempfile.mktemp 安全漏洞 |
| 🟠 **高** | 8 | 裸 except Exception、零覆盖率模块、py.typed 缺失、文档死链 |
| 🟡 **中** | 12 | pH 参数传递、魔法数字、重复代码块、共识评分权重 |
| 🟢 **低** | 15 | 日志不一致、性能热点、emoji 编码、缺失 __all__ |

---

## 1. 严重问题（🔴 立即修复）

### 1.1 评分失败：6 个靶点 PLIP 交互指纹提取崩溃

**位置**: `autodock/interactions.py` → `extract_prolif_fingerprint()` / `extract_plip_interactions()`  
**受影响靶点**: `1B9S`, `2HU4`, `1GWX`, `1F0R`, `1H1P`, `1H22`  
**现象**: 这些靶点的 best-RMSD < 2.0 Å（几何正确），但 top-1 > 2.0 Å（评分失败）。级联分析显示 PLIP/ProLIF 在提取交互指纹时崩溃，导致 IFP 重排失败。  
**根因**: 配体 PDB 文件写入时原子名称或连接表格式不符合 PLIP 解析器预期，引发 `KeyError` 或 `AttributeError`。  
**修复**: 
1. 在 `interactions.py` 中添加更严格的配体 PDB 验证
2. 为 PLIP 崩溃添加特定异常捕获（而非裸 `except Exception`）
3. 当 IFP 提取失败时，回退到纯 Vina 能量评分（当前已部分实现，但需确保不传播 NaN）

**验证**: 运行 `python -m autodock.benchmark --targets 1B9S,2HU4,1GWX,1F0R,1H1P,1H22 --verbose` 并检查 `interactions.py` 的日志输出。

---

### 1.2 MM-GBSA 模板不匹配：3 个靶点无法完成三级重排

**位置**: `autodock/rescoring.py` → `_run_mmgbsa_rescoring()`  
**受影响靶点**: `3EL8`, `1C9K`, `1H1P`  
**现象**: `ValueError: No template found for residue ...` 或 `H atom naming mismatch`。  
**根因**: PDBFixer 添加的氢原子命名（如 `H1`, `H2`, `H3`）与 AMBER 力场模板（`HA`, `HB2`, `HB3` 等）不兼容。这是 **上游限制**（PDBFixer 不保证 AMBER 兼容性）。  
**修复选项**（按优先级）：
1. **短期**: 在 MM-GBSA 失败时优雅降级，记录 `mmgbsa_success=False` 并继续使用 Vina+IFP 结果
2. **中期**: 在受体准备阶段添加 `pdb4amber` + `tleap` 路径作为可选后端（需要 AmberTools 依赖）
3. **长期**: 使用 OpenFF 全参数化路径绕过 AMBER 模板（OpenFF 不依赖残基模板）

**验证**: 运行 `python -m autodock.rescoring --receptor 3EL8_receptor.pdb --ligand 3EL8_ligand.pdbqt` 并确认错误类型。

---

### 1.3 `tempfile.mktemp()` 安全漏洞（CVE 经典竞态条件）

**位置**: 
- `autodock/preparation.py:3466` (`out_path = tempfile.mktemp(suffix="_complex_poseview.pdb")`)
- `autodock/preparation.py:3600` (`output_svg = tempfile.mktemp(suffix="_poseview.svg")`)
- `autodock/interactions.py:983` (`pose_path = tempfile.mktemp(suffix="_pose.pdbqt")`)

**问题**: `tempfile.mktemp()` 自 Python 2.3 起已弃用。它在文件名创建和文件打开之间存在竞态窗口，可被利用进行符号链接攻击或文件覆盖。  
**修复**: 全部替换为 `tempfile.NamedTemporaryFile(delete=False, suffix=...)` 或 `tempfile.mkstemp()`。

```python
# 修复前（危险）
out_path = tempfile.mktemp(suffix="_complex_poseview.pdb")

# 修复后（安全）
fd, out_path = tempfile.mkstemp(suffix="_complex_poseview.pdb")
os.close(fd)  # 立即关闭文件描述符，由调用方重新打开
```

---

## 2. 高优先级问题（🟠 本周修复）

### 2.1 `md_simulation.py` 中 10 处裸 `except Exception`

**位置**: `autodock/md_simulation.py:443-574`（`analyze_md_trajectory()` 函数）  
**问题**: 每个分析子步骤（RMSD、RMSF、PCA、聚类、氢键、接触图）都包裹在 `except Exception as exc: logger.warning(...)` 中。这会掩盖：
- `MemoryError`（应传播，让用户知道内存不足）
- `KeyboardInterrupt`（应传播，让用户能中断运行）
- 库 API 变更（如 MDAnalysis 新版本移除旧 API，应抛出 `ImportError` 或 `AttributeError`）

**修复**: 为每个子步骤捕获特定异常：

```python
# 修复前
except Exception as exc:
    logger.warning(f"Ligand RMSD analysis failed: {exc}")

# 修复后
except (ImportError, ValueError, AnalysisError) as exc:
    logger.warning(f"Ligand RMSD analysis failed: {exc}")
```

**注意**: `MemoryError` 和 `KeyboardInterrupt` 不应被捕获。

---

### 2.2 零覆盖率科学模块

| 模块 | 覆盖率 | 缺失测试 | 风险 |
|------|--------|----------|------|
| `minimization.py` | **0%** | 无 `test_minimization.py` | OpenMM 复合物最小化完全未测试 |
| `alphafold_tools.py` | **6%** | 无 `test_alphafold_tools.py` | AlphaFold 检测启发式未验证 |
| `heatmap.py` | **9%** | 无 `test_heatmap.py` | 交互热图生成未测试 |
| `pipeline.py` | **46%** | 无 `test_pipeline.py` | 端到端管道未测试 |

**修复**: 
1. 为 `minimization.py` 添加至少一个 mocked 测试（mock PDBFixer 和 OpenMM）
2. 为 `alphafold_tools.py` 添加 B-factor 检测单元测试
3. 为 `heatmap.py` 添加小型配体-受体对的交互计数测试
4. 为 `pipeline.py` 添加端到端冒烟测试（使用 `1D4K` 等小型靶点）

---

### 2.3 缺失 `py.typed` 标记

**位置**: `autodock/` 目录  
**问题**: `pyproject.toml` 启用了 `disallow_untyped_defs = true`，但没有 `py.typed` 文件。下游用户的 `mypy` 会**忽略本包的所有类型提示**，导致类型检查失效。  
**修复**: 

```bash
touch autodock/py.typed
```

并确保 `pyproject.toml` 的 `[tool.setuptools.package-data]` 包含 `autodock/py.typed`。

---

### 2.4 README 死链

**位置**: `README.md`  
**问题**: 
- 引用 `METHODS.md`（不存在）
- 引用 `docs/tutorials/`（目录不存在）
- 引用 `yourorg/autodock`（应为 `rongtianhua/autodock`）

**修复**: 
1. 创建 `METHODS.md`（或移除引用）
2. 创建 `docs/tutorials/` 目录并添加至少一个入门教程
3. 全局替换 `yourorg` → `rongtianhua`

---

### 2.5 无真实 Vina 集成测试

**位置**: `autodock/tests/test_docking.py`  
**问题**: 所有对接测试都 mock `vina.Vina`。真实的子进程调用路径（`_run_vina_dock()` 中的 `subprocess` 调用）完全未测试。  
**修复**: 添加 `requires_vina` 标记的集成测试，使用微型受体/配体对（如 `1D4K` 的 5 残基片段 + 苯环配体），验证：
- 子进程正确启动
- PDBQT 输出可解析
- 超时机制工作
- 种子确定性

---

### 2.6 `benchmark.py` 中硬编码 `2.0` 阈值

**位置**: `autodock/benchmark.py:445`, `488`, `489`  
**问题**: 使用字面量 `2.0` 而非导入的 `REDocking_RMSD_THRESHOLD`。如果常量被修改，日志输出将不一致。  
**修复**: 

```python
# 修复前
r["best_rmsd"] <= 2.0

# 修复后
r["best_rmsd"] <= REDocking_RMSD_THRESHOLD
```

---

### 2.7 `StructureCache` 非原子写入

**位置**: `autodock/fetchers.py`  
**问题**: 两个并行获取相同 PDB ID 的进程可能交错写入缓存文件，导致损坏的 PDB/mmCIF。  
**修复**: 使用临时文件 + 原子重命名：

```python
# 修复后
tmp_path = cache_path + ".tmp"
with open(tmp_path, "wb") as fh:
    fh.write(content)
os.replace(tmp_path, cache_path)  # 原子重命名
```

---

### 2.8 宽泛异常元组 `(OSError, RuntimeError, ValueError, TypeError, ImportError)`

**位置**: >15 处，横跨 `preparation.py`, `docking.py`, `validation.py`  
**问题**: 相同的 5 异常类型被复制粘贴到多个调用点。这会掩盖编程错误（如错误的参数类型导致的 `TypeError`），使其被静默捕获为"回退成功"。  
**修复**: 每个调用点只捕获该操作预期的特定异常。例如：

```python
# 修复前
except (OSError, RuntimeError, ValueError, TypeError, ImportError):

# 修复后（文件读取）
except (OSError, ValueError):

# 修复后（库导入）
except ImportError:

# 修复后（数值计算）
except (RuntimeError, ValueError):
```

---

## 3. 中优先级问题（🟡 本月修复）

### 3.1 pH 参数传递已修复，但需验证端到端一致性

**状态**: ✅ **已修复**  
**验证**: `_minimize_complex()` 现在接受 `ph: float = 7.4`，与受体准备的默认值一致。  
**待办**: 运行完整基准测试，确认 `3EL8`, `1C9K`, `1H1P` 的 MM-GBSA 失败不是由 pH 不一致引起。

---

### 3.2 多构象能量数组丢失 Vina 5 分量分解

**位置**: `autodock/docking.py`, `dock_ligand_multi_conformer()`  
**问题**: Vina 返回每个构象的 5 个能量分量（inter, intra, torsions 等），但多构象路径合成假数组：

```python
all_energies = np.array([[e, 0.0, 0.0, 0.0, 0.0] for e, _ in all_poses_pool])
```

**影响**: 下游分析期望有效 Vina 能量分解时将看到零值。  
**修复**: 传播真实能量分量，或在文档中明确标注此限制。

---

### 3.3 穷尽性缩放对大型配体过于激进

**位置**: `autodock/docking.py`, `_auto_exhaustiveness()`  
**问题**: >55 重原子时，exhaustiveness 降至 `base // 8`（最小 4）。base=32 时变为 4——搜索空间缩减 8 倍。  
**修复**: 将下限提高到 8，或添加 `min_exhaustiveness` 配置参数。

---

### 3.4 共识评分仅使用中位数，无权重

**位置**: `autodock/docking.py`, `_consensus_score()`  
**问题**: 简单中位数聚合，无权重或异常值剔除。Vina 能量、IFP 相似度、MM-GBSA 对最终排名的贡献相等。  
**建议**: 考虑加权融合（Vina 40% + IFP 40% + MM-GBSA 20%），或基于基准表现动态学习权重。

---

### 3.5 口袋检测在 P2Rank 失败时中止

**位置**: `autodock/preparation.py`, `find_top_pockets()`  
**问题**: 如果 P2Rank 失败，即使 fpocket 可能成功，也会抛出 `PreparationError`。  
**修复**: 将 fpocket 作为 P2Rank 失败时的自动回退。

---

### 3.6 SDF 输入跳过互变异构体/立体化学枚举

**位置**: `autodock/preparation.py`, `prepare_ligand_from_sdf()`  
**问题**: 硬编码 `molscrub_states=False` 和 `enumerate_stereo=False`。具有未指定立体化学的 SDF 将按原样使用。  
**修复**: 添加 `enumerate_tautomers: bool = False` 和 `enumerate_stereo: bool = False` 参数，让用户选择是否启用。

---

### 3.7 `extract_ligand_from_pdb()` 仅保留最大片段

**位置**: `autodock/utils.py`  
**问题**: 静默丢弃共价加合物、辅因子-抑制剂复合物或盐片段。  
**修复**: 添加 `keep_all_fragments: bool = False` 参数，并在丢弃片段时记录警告日志。

---

### 3.8 无共价抑制剂处理

**位置**: 整个配体准备和对接流程  
**问题**: 共价弹头被当作可逆结合剂处理。无反应基团检测、共价评分或弹头特定准备。  
**修复**: 添加 `--covalent` 标志和反应基团检测（如丙烯酰胺、乙烯基砜、环氧基）。

---

### 3.9 Gasteiger 电荷验证缺失 `inf` 检查

**位置**: `autodock/preparation.py`, `_has_nan_charges()`  
**问题**: 仅检查 `c != c`（NaN）。Gasteiger 电荷在病态分子上也可能产生 `inf`。  
**修复**: 

```python
import math
if not math.isfinite(c):
    return True
```

---

### 3.10 特征值裁剪在口袋形状描述符中人为制造球形

**位置**: `autodock/preparation.py`, `_compute_pocket_shape_descriptors()`  
**问题**: `eigvals = np.maximum(eigvals, 1e-12)` 是任意值。扁平口袋获得人为的球形度。  
**影响**: 仅信息性——不用于评分。  
**修复**: 添加注释说明此裁剪仅为避免 `log(0)`，或改用 `np.maximum(eigvals, eps)` 其中 `eps` 与口袋尺寸相关。

---

### 3.11 MD 积分器平台依赖性数值差异

**位置**: `autodock/md_simulation.py`  
**问题**: 自动选择 Metal → OpenCL → CUDA → CPU。不同平台可能给出略有不同的轨迹。  
**修复**: 记录所选平台，并在 `run_md_stability()` 的返回元数据中添加 `platform` 字段。

---

### 3.12 虚拟筛选种子溢出（32 位限制）

**位置**: `autodock/docking.py`, `virtual_screen()`  
**问题**: 每个化合物的种子为 `base_seed + idx`。对于 >2B 化合物的库，这会溢出 32 位 Vina 种子限制。`validate_seed()` 限制单个种子，但不限制总和。  
**修复**: 使用 `idx % 2_147_483_647` 包装，或改用哈希种子。

---

## 4. 低优先级问题（🟢 打磨）

### 4.1 缺失 `__all__` 在 `__init__.py`

**位置**: `autodock/__init__.py`  
**问题**: `from autodock import *` 导入所有内容，包括内部辅助函数。  
**修复**: 添加 `__all__ = [...]` 显式导出公共 API。

---

### 4.2 `__import__("autodock").__version__` 循环导入风险

**位置**: `autodock/core.py`  
**问题**: 脆弱的循环导入风险。  
**修复**: 使用 `importlib.metadata.version("autodock")`。

---

### 4.3 `safe_subprocess` 截断 stderr 至 300 字符

**位置**: `autodock/core.py:685-690`  
**问题**: `stderr = stderr[-300:]` 静默丢弃长错误消息的前部。Vina 通常在真正错误前发出多行参数警告。  
**修复**: 在截断前将完整 stderr 记录到 DEBUG，或将限制提高到 2000 字符。

---

### 4.4 `_get_vina_seed` 对 `None` 返回确定性种子

**位置**: `autodock/core.py:143-150`  
**问题**: 当 `seed=None` 时，函数返回 `DEFAULT_SEED`（42）。这有文档记录，但**令人惊讶**；大多数用户期望 `None` → 随机种子。  
**修复**: 要么更改行为，要么在使用确定性回退时添加显式警告日志。

---

### 4.5 硬编码 macOS PyMOL 路径

**位置**: `autodock/core.py:228`  
**问题**: `find_pymol()` 硬编码 `/Applications/PyMOL.app/Contents/MacOS/PyMOL`。在 Linux/Windows 上这是死代码。  
**修复**: 移动到 `PATHS` 配置字典或环境变量。

---

### 4.6 Emoji 在 CLI 输出中

**位置**: `autodock/cli.py`  
**问题**: 使用 `⚠️`, `🔍` 等。在 Windows CMD 和某些 CI 日志中显示为乱码。  
**修复**: 添加 ASCII 回退或检测终端编码。

---

### 4.7 VDW 半径表缺少内联引用

**位置**: `autodock/validation.py`  
**问题**: VdW 字典（Bondi 近似）在科学上是合理的，但没有内联引用。  
**修复**: 添加 `# Bondi 1964` 注释。

---

### 4.8 `DockingResult.__post_init__` 缺少长度验证

**位置**: `autodock/core.py`  
**问题**: `center` 和 `box_size` 被强制转换为元组，但不检查长度是否为 3。格式错误的输入将静默传播。  
**修复**: 在 `__post_init__` 中添加 `len(center) == 3` 和 `len(box_size) == 3` 验证。

---

### 4.9 性能：`compute_clash_score` 是 O(N×M)

**位置**: `autodock/validation.py`  
**问题**: 无空间索引（kd-tree 或网格）。对单个构象可接受，但对大型集合较慢。  
**修复**: 使用 `scipy.spatial.cKDTree` 或 `MDAnalysis.lib.distances`。

---

### 4.10 `set_log_level` 仅调整 StreamHandlers

**位置**: `autodock/core.py`  
**问题**: 如果用户添加自定义 FileHandler，其级别不受影响。  
**修复**: 在文档中说明此行为，或递归调整所有处理器。

---

### 4.11 PDB 元素回退可能错误分配双字母元素

**位置**: `autodock/utils.py`, `_read_pdb_atoms_impl()`  
**问题**: 回退取原子名称的第一个字符。"CL" → "C", "BR" → "B", "FE" → "F"。  
**修复**: 在回退中处理双字母名称（检查首字符 + 第二个字符是否形成已知元素符号）。

---

### 4.12 mmCIF 丢失生物组装信息

**位置**: `autodock/utils.py`, `cif_to_pdb_string()`  
**问题**: `gemmi.make_pdb_string()` 转换不对称单元。功能多聚体可能丢失。  
**修复**: 添加注释说明此限制，或支持 `pdbx_struct_assembly` 解析。

---

### 4.13 MODEL/ENDMDL 嵌套未验证

**位置**: `autodock/clustering.py`, `_parse_pose_to_mol()`  
**问题**: 剥离 MODEL/ENDMDL 标签而不验证平衡。  
**修复**: 添加计数器验证 MODEL 和 ENDMDL 数量匹配。

---

### 4.14 `_NON_LIGAND_HETS` 大小写不一致

**位置**: `autodock/benchmark.py`  
**问题**: 使用 `"Fru"`（句首大写）与 `"RIB"`（全大写）并存。  
**修复**: 标准化为全大写 PDB 残基名称。

---

### 4.15 `md_simulation.py` 中 `n_steps` → `ns` 转换使用硬编码 500000

**位置**: `autodock/md_simulation.py`  
**问题**: 硬编码 `500000` 步 = 10 ns（假设 2 fs 步长）。  
**修复**: 添加命名常量 `STEPS_PER_NS = 500_000`。

---

## 5. 已修复问题验证（✅）

| 问题 | 文件 | 修复内容 | 验证状态 |
|------|------|----------|----------|
| Kabsch RMSD 旋转矩阵 | `clustering.py` | `R = Vt.T @ U.T` → `R = U @ Vt` | ✅ 90° 旋转测试通过 |
| MM-GBSA NaN 级联 | `rescoring.py` | 4 层电荷处理（RemoveHs、Gasteiger→formal_charge 回退、`_perturb_zero_charges`、`molecules=` 显式参数） | ✅ 17/20 靶点通过 |
| CI 矩阵扩展 | `.github/workflows/ci.yml` | 添加 macOS + Python 3.10/3.11/3.12 | ✅ 全部绿色 |
| `mp_context="spawn"` | `docking.py` | 所有 `ProcessPoolExecutor` 调用添加 `mp_context=mp_ctx` | ✅ 代码审查确认 |
| pH 参数传递 | `minimization.py` | `_minimize_complex()` 添加 `ph: float = 7.4` | ✅ 代码审查确认 |
| MD 积分器种子 | `md_simulation.py` | `integrator.setRandomNumberSeed(seed)` | ✅ 代码审查确认 |
| 坐标回退 RMSD | `validation.py` | `compute_rmsd_to_crystal()` 添加坐标回退 | ✅ 代码审查确认 |

---

## 6. 模块健康评分（更新）

| 模块 | 行数 | 评级 | 关键关切 | 覆盖率 |
|------|------|------|----------|--------|
| `core.py` | 815 | A- | 小 docstring 缺口，`__import__` 反模式 | ~85% |
| `docking.py` | 1,673 | B+ | 重复代码块，魔法数字，temp-file 泄漏 | ~78% |
| `preparation.py` | 4,325 | B | 过大；重复残基重命名；需要重构为子模块 | ~72% |
| `validation.py` | 827 | A- | VdW 表引用，O(N×M) 冲突评分 | ~80% |
| `benchmark.py` | 751 | B+ | 硬编码阈值字面量，宽泛 except 元组 | ~75% |
| `cli.py` | 1,215 | B | Emoji 编码，文档缺失时死 `--help` 文本 | ~60% |
| `minimization.py` | 533 | B+ | Temp-file 泄漏，Kabsch 对齐正确但脆弱 | **0%** |
| `md_simulation.py` | 584 | C+ | **10 处裸 `except Exception`** — 代码库中风险最高 | ~45% |
| `rescoring.py` | 420 | B | MM-GBSA 模板不匹配（上游限制） | ~55% |
| `interactions.py` | 680 | B- | PLIP 崩溃导致 6 个评分失败 | ~50% |
| `fetchers.py` | 380 | B | `StructureCache` 非原子写入 | ~65% |
| `utils.py` | 1,200 | B+ | PDB 元素回退，mmCIF 生物组装 | ~70% |
| `alphafold_tools.py` | 200 | C | 无测试，B-factor 启发式未验证 | **6%** |
| `heatmap.py` | 180 | C | 无测试 | **9%** |

---

## 7. 与官方工具文档的对齐情况

| 工具 | 版本要求 | 对齐状态 | 偏差 |
|------|----------|----------|------|
| AutoDock Vina | ≥1.2 | ✅ | 默认 `exhaustiveness=32`, `n_poses=20` 符合发表标准 |
| RDKit | ≥2023.03 | ✅ | ETKDGv3 用于构象生成，GetBestRMS 用于 RMSD |
| Meeko | ≥0.6 | ✅ | `allow_bad_res=True` 记录警告 |
| OpenMM | ≥8.1 | ✅ | 单位转换正确（Quantity 对象自动转换） |
| OpenFF | ≥2.0 | ✅ | 小分子力场参数化 |
| P2Rank | ≥2.4 | ⚠️ | 失败时无 fpocket 回退 |
| PLIP | ≥2.3 | ⚠️ | 配体 PDB 格式偶尔不兼容 |
| ProLIF | ≥2.0 | ⚠️ | 同上 |
| PoseBusters | ≥0.2 | ✅ | 化学合理性检查 |
| PDBFixer | 最新 | ⚠️ | H 原子命名与 AMBER 模板不兼容（上游限制） |

---

## 8. 顶刊发表标准符合度

### 8.1 方法学透明度（✅ 良好）

- ✅ 所有默认参数文献支持（Vina 32/20/3.0，RMSD 2.0 Å，冲突 1.2 Å）
- ✅ `HARD_TARGET_OVERRIDES` 带内联 `_note` 字段，优秀科学文档
- ✅ 基准测试 20 靶点覆盖激酶、蛋白酶、核受体、酶
- ✅ 三级级联重排（Vina → IFP → MM-GBSA）有明确文档

### 8.2 可复现性（⚠️ 部分）

- ✅ 默认种子确定性（`seed=42`）
- ✅ 虚拟筛选每化合物种子确定性
- ✅ 系综对接每重复种子确定性
- ⚠️ MD 轨迹因平台选择（Metal/OpenCL/CUDA/CPU）而有数值差异
- ⚠️ 无 `py.typed`，下游类型检查失效

### 8.3 代码质量（⚠️ 需改进）

- ✅ ruff + black + pytest CI
- ✅ 类型提示覆盖公共 API
- ❌ 零覆盖率模块（minimization.py, alphafold_tools.py, heatmap.py）
- ❌ 无真实 Vina 集成测试
- ❌ 裸 `except Exception` 掩盖错误

### 8.4 文档完整性（❌ 不足）

- ❌ README 死链（METHODS.md, docs/tutorials/）
- ❌ 无 CHANGELOG.md（虽然存在但可能不完整）
- ❌ 无贡献指南（虽然 CONTRIBUTING.md 存在）
- ❌ 教程缺失

### 8.5 许可证与元数据（⚠️ 部分）

- ✅ LICENSE 文件存在（MIT）
- ✅ CITATION.cff 存在
- ❌ `py.typed` 缺失
- ❌ README badge 指向 `yourorg` 而非 `rongtianhua`

---

## 9. 推荐行动计划（优先级排序）

### 阶段 1：安全与稳定性（本周）

1. **替换所有 `tempfile.mktemp()`** → `mkstemp()`（3 处）
2. **修复 `md_simulation.py` 裸 `except Exception`** → 特定异常（10 处）
3. **修复 `StructureCache` 非原子写入** → 临时文件 + `os.replace`
4. **添加 `py.typed`** → `touch autodock/py.typed`
5. **修复 README 死链** → 创建 METHODS.md 和 docs/tutorials/

### 阶段 2：科学正确性（下周）

6. **调试 6 个评分失败靶点** → 检查 `interactions.py` PLIP 崩溃日志
7. **处理 3 个 MM-GBSA 模板不匹配** → 添加 `pdb4amber` 路径或优雅降级
8. **统一 `benchmark.py` 硬编码 `2.0`** → 使用 `REDocking_RMSD_THRESHOLD`
9. **收窄宽泛异常元组** → 每调用点特定异常
10. **添加 `inf` 检查到 Gasteiger 验证** → `math.isfinite(c)`

### 阶段 3：测试加固（下月）

11. **创建 `test_minimization.py`**（即使 mocked）
12. **创建 `test_alphafold_tools.py`**（B-factor 检测单元测试）
13. **创建 `test_heatmap.py`**（小型交互计数测试）
14. **添加真实 Vina 集成测试**（`requires_vina` 标记）
15. **提高覆盖率 fail-under 至 50%**（当前 60%，但 minimization.py 为 0%）

### 阶段 4：工程打磨（持续）

16. **提取重复代码** → MODEL/ENDMDL 剥离、残基重命名集中到 `utils.py`
17. **添加命名常量** → `_auto_exhaustiveness` 阈值、VDW 引用
18. **重构 `preparation.py`** → 拆分为 `receptor_prep.py`, `ligand_prep.py`, `pocket_detection.py`
19. **添加空间索引到冲突检测** → `cKDTree`
20. **验证 `DockingResult` 字段长度** → `__post_init__` 中 `len(center) == 3`

---

## 10. 基准测试现状与目标

| 指标 | 当前 | 目标 | 差距 |
|------|------|------|------|
| Top-1 RMSD ≤ 2.0 Å | 55% (11/20) | 70% (14/20) | +3 靶点 |
| Best-achievable | 85% (17/20) | 90% (18/20) | +1 靶点 |
| 评分失败 | 6 靶点 | 0 靶点 | -6 靶点 |
| MM-GBSA 功能 | 17/20 | 20/20 | +3 靶点 |
| 级联救援 | 4/20 (Tier-2) | 6/20 | +2 靶点 |

**关键路径**: 修复 6 个评分失败 → top-1 可能提升至 65-70%。修复 3 个 MM-GBSA 不匹配 → 三级级联完全功能化。

---

*报告生成时间*: 2026-06-08  
*审计方法*: 静态分析、模式搜索、关键函数逐行审查、与 pyproject.toml 和 README 交叉引用、与历史审计报告对比  
*建议复查周期*: 每 2 周或每次重大提交后
