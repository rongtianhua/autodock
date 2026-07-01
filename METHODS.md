# METHODS.md — autodock 方法学文档

本文档描述 `autodock` 管线默认使用的方法、参数与科学假设，供复现、发表与审计使用。

---

## 1. 概述

`autodock` 是一套基于 **AutoDock Vina** 的端到端分子对接管线，覆盖结构准备、口袋检测、分子对接、构象聚类、相互作用分析、几何验证、能量最小化与报告生成。默认参数以**发表级可重复性**为目标，所有随机步骤均使用固定种子（`seed=42`）。

### 1.1 默认参数总览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ph` | 7.4 | 氢原子添加与质子化状态 |
| `exhaustiveness` | 32 | Vina 搜索力度（单配体 / 虚拟筛选一致） |
| `n_poses` | 20 | 输出 pose 数量 |
| `seed` | 42 | 随机数种子 |
| `energy_range` | 3 | 保留 pose 与最优 pose 的能量差上限（kcal/mol） |
| `min_rmsd` | 1.0 | Vina pose 间最小 RMSD（Å） |
| `box_padding` | 5.0 | 口袋边界框额外填充（Å） |
| `rmsd_threshold` | 2.0 | pose 聚类 / redocking 成功阈值（Å） |
| `md_steps` | 50000 | 短 MD 模拟步数（100 ps @ 2 fs） |
| `md_temperature` | 300 K | MD 模拟温度 |

> **注意**：自 2026-07-01 起，虚拟筛选默认值已从 `exhaustiveness=16, n_poses=3` 上调为 `32/20`，与单配体对接一致。若需高通量筛选，可显式降低参数并在日志中声明。

---

## 2. 受体准备

受体准备由 `prepare_receptor()` 执行，默认六步流程：

1. **输入规范化**：
   - 支持 PDB / mmCIF / PDBFixer 输出；mmCIF 通过 `gemmi` 转换为 PDB。
   - 多链受体默认按 `receptor_multichain_strategy` 处理：
     - `extract_single`：保留最长链（默认，适用于大多数单体体系）。
     - `multichain`：保留全部链（适用于多聚体界面口袋）。

2. **PDBFixer 修补**（`openmm`）：
   - 查找并替换非标准残基。
   - 查找并填充缺失重原子（`addMissingAtoms`）。
   - 在 `ph=7.4` 下添加缺失氢原子（`addMissingHydrogens`）。
   - 不删除异源分子（HETATM）—— 功能水与金属离子由后续步骤保留。

3. **功能水与金属离子保留**：
   - 金属离子（Zn²⁺、Mg²⁺、Ca²⁺、Mn²⁺、Fe²⁺/Fe³⁺、Cu²⁺、Co²⁺、Ni²⁺）默认保留。
   - 与金属配位的水分子（距离 ≤ 2.5 Å）或与配体结合位点水分子保留。
   - 其余结晶水默认删除，以减少搜索噪音。

4. **氢原子优化**（`reduce` 或 PDB2PQR）：
   - 优先使用 `reduce` 优化氢原子位置（Asn/Gln/His 翻转）。
   - 若 `reduce` 不可用，回退至 PDB2PQR + PROPKA。

5. **PDBQT 生成**（`Meeko`）：
   - 受体使用 `meeko.Polymer` 生成 PDBQT，保留电荷与原子类型。
   - 若 Meeko 失败，回退至 Open Babel。

6. **可选 OpenMM 最小化**：
   - 默认关闭；启用时对整个受体进行 100 步约束最小化（约束重原子，k=500 kJ/mol/nm²）。

---

## 3. 配体准备

配体准备由 `prepare_ligand()` / `prepare_ligand_from_smiles()` 执行：

1. **输入解析**：
   - SMILES 字符串或 SDF 文件。
   - 从 SMILES 时，使用 RDKit `EmbedMolecule` + ETKDGv3 生成 3D 构象。
   - 从 SDF 时，保留输入 3D 坐标，不枚举互变异构体/立体异构体。

2. **质子化**：
   - 使用 `molscrub` 生成主要质子化状态（若安装）。
   - 否则使用 RDKit 在 `ph=7.4` 下进行简单质子化估计。

3. **PDBQT 生成**（`Meeko`）：
   - 使用 `meeko.MoleculePreparation` 生成 Gasteiger-Marsili 电荷与 AutoDock 原子类型。
   - 检查 `inf` 电荷并替换为 0.0，避免 Vina 崩溃。
   - 若 Meeko 失败，回退至 Open Babel。

4. **多构象模式**（`multi_conformer=True`）：
   - 使用 RDKit `EnumerateMolFromMolBlock`/`EmbedMultipleConfs` 生成 `n_conformers`（默认 10）。
   - 每个构象独立对接，结果合并后聚类。

---

## 4. 口袋检测

口袋检测由 `find_top_pockets()` 执行，采用 **P2Rank 主路径 + fpocket 交叉验证** 的级联策略：

1. **P2Rank**（若可用）：
   - 运行 `p2rank predict`，输出每个口袋的概率 `prob`。
   - 解析 `_predictions.csv` 获取口袋中心、体积、残基 ID。
   - 仅保留 `prob ≥ 0.5` 的口袋。

2. **fpocket 验证**（若可用）：
   - 对同一受体运行 `fpocket`，解析 `_info.txt` 获取 α-sphere 口袋。
   - 当 P2Rank 口袋与 fpocket 口袋中心距离 ≤ 4 Å 时，标记为 `verified`。
   - 验证口袋优先采用 fpocket 的几何中心与可药性评分。

3. **边界框计算**：
   - 对每个口袋，使用 `box_size = max(dim_x, dim_y, dim_z) + 2 * box_padding`。
   - 默认最小边界框 20 Å，最大 40 Å。

4. **已知活性位点（可选）**：
   - 若提供 `known_active_site`，优先选择最近口袋并标记 `pocket_type`（orthosteric / allosteric）。

---

## 5. 分子对接

对接由 `dock_ligand()` 调用 AutoDock Vina（CLI 或 Python API）执行：

1. **输入**：受体 PDBQT、配体 PDBQT、口袋中心 `center`、边界框 `box_size`。
2. **Vina 参数**：
   - `exhaustiveness=32`
   - `n_poses=20`
   - `energy_range=3`
   - `min_rmsd=1.0`
   - `seed=42`
3. **自动降采样**（大配体）：
   - 当可旋转键数 > 15 时，自动降低 `exhaustiveness`（下限 16）以控制计算成本。
4. **输出**：
   - `best_pose.pdbqt`：最优 pose（按亲和力）。
   - `all_poses.pdbqt`：全部 pose。
   - `DockingResult` 数据类包含亲和力、聚类、受体来源等信息。

### 5.1 构象聚类

对接后使用 `autodock.clustering.cluster_poses()` 对 pose 进行基于 RMSD 的凝聚聚类：

- 阈值：2.0 Å。
- 每个簇的代表 pose 为能量最低的成员。
- 最多保留前 5 个簇的代表 pose。

### 5.2 共识评分

最佳 pose 使用 `_consensus_score()` 进行多打分函数重评：

- 主打分函数：`vina`（默认）或 `vinardo`。
- 可用时同时计算 `vinardo` 与 `ad4` 映射分。
- 共识亲和力取可用打分函数的中位数。

> 当前共识为简单中位数，无权重或异常值剔除，结果应作为辅助参考。

---

## 6. 验证

### 6.1 PoseBusters 几何检查

使用 `PoseBusters(config="dock")` 对最佳 pose 进行物理合理性检查：

- 默认排除与 Vina 化学模型冲突的项：
  - `non-aromatic_ring_non-flatness`（Vina pose 保留 puckered 环构象）。
  - `mol_pred_loaded` / `mol_true_loaded` / `mol_cond_loaded`（文件加载标志）。
  - `minimum_distance_to_organic_cofactors` / `minimum_distance_to_inorganic_cofactors`。
- 必须通过的项包括：无原子重叠、键长/键角/扭转角合理、手性一致等。

### 6.2 RMSD 计算

- 主路径：RDKit `GetBestRMS`（拓扑对齐，考虑对称性）。
- 回退：Kabsch 坐标对齐 + 重原子坐标 RMSD（`validation.compute_rmsd_coordinate_based`）。
- Redocking 成功标准：最佳 pose RMSD ≤ 2.0 Å。

### 6.3 Clash 检测

- 计算配体重原子与受体重原子间距离 < 2.0 Å 的冲突数。
- 输出 `clash_score`（每原子平均冲突数）。

---

## 7. 相互作用分析

相互作用分析由 `interactions.py` 执行：

1. **主路径：PLIP**（若安装）：
   - 识别氢键、疏水作用、π-π、π-阳离子、盐桥、卤键、水桥、金属配位。
   - 输出相互作用指纹（IFP）。
2. **交叉验证：ProLIF**（若安装）：
   - 作为 PLIP 结果的交叉验证，特别用于 IFP 相似性比较。
3. **相互作用指纹（IFP）**：
   - 将受体残基-配体相互作用编码为二进制向量。
   - 用于 pose 重排与聚类。

---

## 8. 后处理

### 8.1 OpenMM 能量最小化

`minimize_ligand()` 对最佳 pose 进行配体约束最小化：

- 使用 OpenFF 分子与 AMBER/ff14SB + 小分子力场。
- 重原子施加位置约束（`restraint_k=500 kJ/mol/nm²`）。
- 默认最大迭代次数 1000。
- 输出最小化后 pose，用于改善 PoseBusters 通过率。

### 8.2 短 MD 稳定性模拟

`run_short_md()` 执行溶剂化短 MD：

- 溶剂：TIP3P-FB 水盒，最小 1.0 nm 边界。
- 离子：添加中性化离子，力场 `amber14-all.xml`。
- 积分器：Langevin Middle Integrator，2 fs 步长，300 K。
- 模拟长度：50,000 步（100 ps）。
- 输出：配体重原子 RMSD、蛋白质骨架 RMSD、关键氢键保留率。

---

## 9. 虚拟筛选与批量对接

### 9.1 虚拟筛选

`virtual_screen()` 对化合物库并行对接：

- 默认参数与单配体一致：`exhaustiveness=32, n_poses=20, seed=42`。
- 支持多 worker（`workers=-1` 使用全部 CPU）。
- 输出 CSV 排名表，含亲和力、共识分、PoseBusters 结果、RMSD（若有晶体配体）。
- 结果按 `best_affinity` 升序（越负越好）。

### 9.2 批量对接

`batch_dock()` 支持多受体 × 多配体矩阵对接，每对独立调用 `dock_ligand()`。

---

## 10. 基准测试

`benchmark.py` 包含 20 靶点 redocking 基准集：

- 靶点类型覆盖激酶、蛋白酶、核受体、酶等。
- 评估指标：
  - **best-achievable**：全部 pose 中最低 RMSD ≤ 2.0 Å 的比例。
  - **top-1**：Vina 排名第一 pose 的 RMSD ≤ 2.0 Å 的比例。
  - **top-3**：前三 pose 中至少一个 RMSD ≤ 2.0 Å 的比例。
- 困难靶点（采样/打分挑战）在 `HARD_TARGET_OVERRIDES` 中配置特殊边界框或增加采样。

当前表现（P0 20 靶点）：

| 指标 | 数值 |
|------|------|
| best-achievable | 70% |
| top-3 | 55% |
| top-1 | ~40% |

---

## 11. 缓存

`autodock.cache` 使用 SHA-256 参数敏感缓存：

- 受体准备结果缓存键：输入文件哈希 + 准备参数。
- 配体准备结果缓存键：SMILES/SDF 哈希 + 准备参数。
- 口袋检测结果缓存键：受体文件哈希 + `padding` + `max_pockets`。
- 缓存目录默认：`~/.autodock/cache`。

---

## 12. 已知限制

- **共价配体**：未检测共价 warhead，共价抑制剂按可逆配体处理。
- **多聚体受体**：默认 `extract_single` 可能丢失链间口袋，建议使用 `multichain` 策略。
- **柔性受体**：Meeko Polymer 柔性侧链支持有限，主要依赖刚性受体对接。
- **SDF 立体化学**：不枚举互变异构体/立体异构体。
- **MM-GBSA**：OpenMM/AMBER 模板可能与 PDBFixer 氢命名不兼容，失败时优雅降级。

---

## 13. 引用

使用本管线发表时，建议引用以下核心工具：

- Trott, O. & Olson, A. J. (2010). AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *J. Comput. Chem.* 31, 455-461.
- RDKit: Open-source cheminformatics. https://www.rdkit.org
- Forli, S. et al. (2016). Computational protein-ligand docking and virtual drug screening with the AutoDock suite. *Nat. Protoc.* 11, 905-919.
- Krivák, R. & Hoksza, D. (2018). P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. *J. Cheminform.* 10, 39.
- Schmidtke, P. et al. (2010). fpocket: online tools for protein ensemble pocket detection and tracking. *Nucleic Acids Res.* 39, W283-W288.
- Salentin, S. et al. (2015). PLIP: fully automated protein-ligand interaction profiler. *Nucleic Acids Res.* 43, W443-W447.
- Gowers, R. J. et al. (2016). MDAnalysis: A Python Package for the Rapid Analysis of Molecular Dynamics Simulations. *Proceedings of the 15th Python in Science Conference*, 98-105.

---

## 14. 环境变量

以下环境变量可用于覆盖自动发现的外部工具路径：

| 变量 | 作用 |
|------|------|
| `P2RANK_HOME` | P2Rank 安装目录；`find_p2rank()` 优先使用 `$P2RANK_HOME/prank`。 |
| `PYMOL_EXE` | PyMOL 可执行文件路径；优先于 Schrödinger / conda 自动发现。 |

## 15. 版本历史

| 日期 | 版本 | 方法学变更 |
|------|------|------------|
| 2026-07-01 | 1.0.0+dev | 虚拟筛选默认参数上调至 `exhaustiveness=32, n_poses=20`；统一临时文件清理规范；`P2RANK_HOME`/`PYMOL_EXE` 环境变量支持。 |
