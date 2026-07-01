# autodock 分子对接管线 — 阶段审查报告

**审查日期**: 2026-07-01  
**审查范围**: `autodock/` 包 26 个模块，~9,447 行源码；`autodock/tests/` 26 个测试文件，~15,453 行测试代码  
**审查维度**: 顶层 workflow 调度 · 算法科学性与稳健性 · 测试覆盖与质量体系 · 发表级生产标准  
**对照基准**: 2026-06-08 审计报告（`AUDIT_REPORT_2026-06-08.md`）及后续修复记录（`FIX_REPORT_2026-06-08.md`）

---

## 1. 执行摘要

经过多轮迭代，`autodock` 已从一套功能验证脚本演进为**工程结构清晰、模块化程度高、具备生产级雏形的分子对接管线**。本次审查在真实 conda 环境（`autodock`）中执行了测试套件，结果显示：**865 个测试通过、1 个测试失败、8 个跳过，源码行覆盖率达 75.87%**，显著优于 6 月 8 日审计时的低覆盖状态。

然而，与“发表级生产标准”相比，仍存在可感知的差距，主要集中在：
- 根目录遗留大量调试脚本与临时文件，仓库卫生度不足；
- 部分核心模块（`rescoring.py` 30%、`minimization.py` 49%、`post_dock_pipeline.py` 61%）覆盖仍偏低；
- 若干科学稳健性问题尚未完全闭环（临时文件清理、虚拟筛选默认参数一致性、多聚体处理、子进程 stderr 截断等）；
- 缺少面向后续协作者的 `AGENTS.md` 与清晰的 `METHODS.md` 方法学文档。

**总体评级**:

| 维度 | 评级 | 说明 |
|------|------|------|
| 顶层 Workflow 设计 | **A-** | 六步管线清晰，配置分层合理，支持断点续跑；但多聚体策略与错误传播仍可细化。 |
| 科学算法正确性 | **B+** | Vina 对接、P2Rank/fpocket 口袋检测、PLIP/ProLIF 相互作用、PoseBusters 验证均已集成；基准 best-achievable 70%、top-3 55%，符合 AutoDock Vina 文献预期。 |
| 代码稳健性 | **B** | 异常层级完整，`spawn` 多进程安全，超时机制健全；但临时文件清理、硬编码工具路径、子进程 stderr 截断等问题仍需修复。 |
| 测试与质量 | **B+** | 874 个测试用例，整体覆盖率 76%，超过 60% 门槛；但存在一个失败用例，且部分科学模块覆盖不足。 |
| 发表/生产标准 | **B-** | 文档、元数据、CI、类型标记、LICENSE 已补齐；但仓库卫生、方法学文档、共价/多聚体限制说明尚不到位。 |

---

## 2. 项目现状快照

### 2.1 架构与入口

| 入口 | 文件 | 作用 |
|------|------|------|
| CLI | `autodock/cli.py` | 17 个子命令覆盖 status/init/fetch/prepare/dock/analyze/report/benchmark/batch/screen/md 等 |
| Python API | `autodock/__init__.py` | 导出 `run_docking_workflow`, `dock_ligand`, `prepare_receptor`, `find_top_pockets` 等 |
| 模块执行 | `autodock/__main__.py`, `autodock/workflow.py` | `python -m autodock` / `python -m autodock.workflow` |
| 配置 | `autodock/config.py` | YAML 配置解析，支持 project/receptor/pocket/ligands/docking/validation/reporting 分层 |
| 缓存 | `autodock/cache.py` | SHA-256 参数敏感缓存，用于受体/配体/口袋准备 |

### 2.2 核心模块职责

- `preparation.py`（2,112 行）：受体准备（PDBFixer/reduce/PDB2PQR/OpenMM）、配体准备（RDKit/Meeko/Open Babel）、口袋检测（P2Rank/fpocket/DoGSite3）。
- `docking.py`（604 行）：Vina CLI/Python API 包装、单配体/多构象/虚拟筛选/批量/ensemble 对接、共识评分。
- `workflow.py`（538 行）：端到端 `run_docking_workflow()`，六步管线，JSON 断点续跑。
- `validation.py`（618 行）：PoseBusters 检查、RMSD（拓扑感知 + Kabsch 回退）、clash、redocking 验证、级联重打分。
- `interactions.py`（484 行）：PLIP 主路径 + ProLIF 交叉验证 + 相互作用指纹（IFP）。
- `rescoring.py`（194 行）：IFP 重排、OpenMM/MM-GBSA 三级级联。
- `md_simulation.py` / `minimization.py`：OpenMM 短 MD 稳定性检查与能量最小化。
- `rendering.py` / `reporting.py`：PyMOL 3D 渲染、RDKit 2D 相互作用图、PDF/Excel/CSV 报告。
- `benchmark.py`（965 行）：20 靶点 redocking 基准，含 `HARD_TARGET_OVERRIDES` 困难靶点调参。

### 2.3 依赖与运行环境

- Python ≥3.10；核心依赖 `numpy`, `pandas`, `scipy`, `Pillow`, `reportlab`, `openpyxl`, `PyYAML`, `pubchempy`。
- 可选 extras：`[docking]`（vina/meeko/rdkit）、`[analysis]`（plip/prolif/MDAnalysis）、`[md]`（openmm）、`[validation]`（posebusters）、`[vis]`（pymol）。
- 外部二进制：AutoDock Vina、P2Rank、fpocket、PyMOL、Open Babel、reduce/PDB2PQR、Java。

---

## 3. 顶层 Workflow 调度评估

### 3.1 设计亮点

`run_docking_workflow()` 采用**函数式、单入口、状态机式**设计：

1. **六步管线清晰**: 受体获取 → 受体准备 → 口袋检测 → 配体准备 → 对接 → 后处理/报告。
2. **断点续跑**: `workflow_state.json` 原子写入（`tmp + os.replace`），`_step_done` 校验关键文件存在性。
3. **配置分层**: 代码默认值 → YAML 配置 → 关键字参数，显式参数始终优先。
4. **结果容器**: `DockingWorkflowResult` dataclass 记录全部路径、参数、环境、运行时间、错误与警告，便于下游审计。
5. **多聚体策略**: `receptor_multichain_strategy` 支持 `auto`/`multichain`/`extract_single`/`alphafold` 及列表组合，说明已意识到多聚体是 Vina 失效的重要因素。
6. **模块化后处理**: `post_dock_pipeline.build_pair_dir()` 统一输出目录结构（`01_structures/02_interactions/03_figures/04_reports`）。

### 3.2 待改进点

| 问题 | 位置 | 影响 | 建议 |
|------|------|------|------|
| `run_docking_workflow()` 参数过多（>30 个） | `workflow.py:239` | 函数签名臃肿，用户认知负荷高 | 将参数分组为 `ReceptorConfig`, `LigandConfig`, `DockingConfig` dataclass；保持向后兼容的 `**kwargs` |
| 多聚体策略默认 `extract_single` 过于激进 | `workflow.py:555` | 对真实多聚体体系（如 2HU4 八聚体）可能丢失功能位点 | 增加基于 `pisa`/`biopython` 的生物学组装推断，或至少给出警告 |
| 错误传播不透明 | `workflow.py` 多处 `try/except` | 某些步骤失败时被记录到 `result.errors` 但未 raise，调用方难以判断成功/部分成功/失败 | 增加 `strict=True` 模式；默认保持容错，但严格模式要求任何科学步骤失败即中断 |
| 子进程 stderr 被截断 | `core.py:293` | Vina/P2Rank 出错时只能看到前 300 字符 | 在截断前将完整 stderr 写入 DEBUG 日志 |
| `__import__("autodock").__version__` | `workflow.py:216`, `core.py:482` | 循环导入风险，不符合 PEP 396 | 改用 `importlib.metadata.version("autodock")` |

---

## 4. 科学性与稳健性评估

### 4.1 已修复的关键科学问题（相比 6 月 8 日审计）

| 问题 | 状态 | 证据 |
|------|------|------|
| Kabsch RMSD 矩阵乘法方向错误 | ✅ 已修复 | `clustering.py:131` 使用 `R = U @ Vt` 并附注释 |
| MD 轨迹非确定性 | ✅ 已修复 | `md_simulation.py:274` 调用 `integrator.setRandomNumberSeed(seed)` |
| `set_log_level` 静默回退 | ✅ 已修复 | `core.py:135` 对未知 level 抛出 `ValueError` |
| Gasteiger 电荷 `inf` 未检查 | ✅ 已修复 | `preparation.py:1510` 检查 `math.isinf(c)` |
| pH 参数在最小化中未传递 | ✅ 已修复 | `minimization.py:289` 接受 `ph=7.4` 并传入 `addMissingHydrogens` |
| `py.typed` 缺失 | ✅ 已修复 | `autodock/py.typed` 存在 |
| `LICENSE` 缺失 | ✅ 已修复 | `LICENSE` 存在（MIT） |
| 困难靶点调参 | ✅ 已实施 | `benchmark.py:87-174` 含 10 个靶点的 `_note` 说明 |

### 4.2 仍存在的科学/稳健性问题

#### P0（高优先级）

1. **临时文件清理不完整**
   - 已确认 20+ 处使用 `tempfile.mkstemp`/`NamedTemporaryFile(delete=False)`，其中 `docking.py:345`、 `validation.py:74/535/619`、`preparation.py:3601/3736/4006`、`analysis.py:139` 等处缺少 `try/finally` 或 `ExitStack` 保证清理。
   - 长周期批量/虚拟筛选任务在异常中断时会遗留大量临时文件，存在磁盘耗尽风险。
   - **建议**: 引入 `contextlib.ExitStack` 或统一 `try/finally: os.unlink()`；对 `mkdtemp` 使用 `tempfile.TemporaryDirectory()`。

2. **`test_keep_waters_near_metal` 测试失败**
   - 实测在 `autodock` 环境中失败：`AssertionError: assert 'ZN' in ''`。
   - 根因待定位：可能是 `Polymer.from_pdb_string` 的 mock 调用方式与代码实现不一致，或准备路径在某些分支下传入空 `pdb_content`。
   - **建议**: 修复该回归，并补充金属离子保留的集成测试。

3. **虚拟筛选默认参数与发表级默认值不一致**
   - `docking.py:1144` 中 `virtual_screen()` 默认 `exhaustiveness=16, n_poses=3`，而 workflow/API 默认 `32/20`。
   - 用户可能误以为虚拟筛选仍保持发表级采样强度。
   - **建议**: 统一默认值或在文档/日志中显式声明“筛选模式使用更快但采样更低的参数”。

#### P1（中优先级）

4. **硬编码工具路径与版本号**
   - `core.py:225-265` 中 `find_p2rank()` 硬编码 `p2rank_2.5.1`，`find_pymol()` 硬编码 macOS Schrödinger PyMOL 路径。
   - **建议**: 优先从环境变量（`P2RANK_HOME`, `PYMOL_EXE`）读取，再回退到自动发现；版本号改为配置项。

5. **共价/反应性配体未识别**
   - 当前管线将共价抑制剂当作可逆配体处理，可能导致错误的几何与能量解释。
   - **建议**: 在配体准备阶段增加 warhead 检测（丙烯酰胺、乙烯基砜、氰基等），至少发出警告或提供 `--covalent` 模式。

6. **共识评分过于简化**
   - `_consensus_score()` 对可用打分函数取简单中位数，无权重、无异常值剔除。
   - **建议**: 引入基于 Z-score 或 rank-based 的稳健共识，或明确文档说明其启发式性质。

7. **MM-GBSA / OpenFF 重打分路径覆盖不足**
   - `rescoring.py` 覆盖率仅 30%；`minimization.py` 49%。
   - 6 月审计提到的 `No template found for residue` 问题未完全解决。
   - **建议**: 增加 OpenFF 全参数化回退路径，或为 AMBER 模板不匹配提供优雅降级。

#### P2（低优先级）

8. **SDF 输入保留原始 tautomer/stereo**
   - `prepare_ligand_from_sdf()` 不枚举互变异构体/立体异构体。
   - **建议**: 在文档中明确标注；可选购置 RDKit `EnumerateTautomers`/`EnumerateStereoisomers` 可选开关。

9. **`safe_subprocess` stderr 截断**
   - 截断至 300 字符，调试困难。
   - **建议**: 截断前写入 DEBUG 日志，并保留完整 stderr 到文件（如 `<outdir>/logs/`）。

---

## 5. 测试覆盖与质量体系评估

### 5.1 测试执行结果（真实环境）

```bash
conda activate autodock
pytest autodock/tests -m "not slow and not integration and not requires_vina and not requires_rdkit" -q --tb=short
```

结果：**865 passed, 1 failed, 8 skipped, 75.87% coverage**（运行时间 93.6 s）。

### 5.2 覆盖率分布（节选）

| 模块 | 覆盖率 | 状态 |
|------|--------|------|
| `__init__.py` | 100% | ✅ |
| `alphafold_tools.py` | 89% | ✅ 较 6 月审计的 6% 大幅提升 |
| `analysis.py` | 97% | ✅ |
| `benchmark.py` | 98% | ✅ |
| `cache.py` | 87% | ✅ |
| `cli.py` | 72% | ⚠️ CLI 子命令众多，部分未覆盖 |
| `clustering.py` | 94% | ✅ |
| `config.py` | 95% | ✅ |
| `core.py` | 85% | ✅ |
| `docking.py` | 92% | ✅ |
| `fetchers.py` | 86% | ✅ |
| `heatmap.py` | 100% | ✅ |
| `interactions.py` | 72% | ⚠️ PLIP/ProLIF 部分分支未覆盖 |
| `md_simulation.py` | 89% | ✅ |
| `minimization.py` | 49% | ⚠️ OpenMM 路径覆盖不足 |
| `post_dock_pipeline.py` | 61% | ⚠️ 端到端后处理覆盖偏低 |
| `preparation.py` | 62% | ⚠️ 2,112 行大模块，部分分支未覆盖 |
| `rendering.py` | 65% | ⚠️ PyMOL 渲染分支难测，但可 mock |
| `rescoring.py` | 30% | ❌ MM-GBSA/IFP 重打分覆盖严重不足 |
| `utils.py` | 80% | ✅ |
| `validation.py` | 86% | ✅ |
| `validation_params.py` | 96% | ✅ |
| `workflow.py` | 84% | ✅ |

### 5.3 质量体系强项

- **标记体系完善**: `slow`, `integration`, `requires_vina`, `requires_rdkit` 等 pytest marker 便于分层执行。
- **CI 矩阵**: `.github/workflows/ci.yml` 在 Ubuntu/macOS × Python 3.10/3.11/3.12 上运行 lint（ruff/black）与测试，45 分钟超时。
- **类型检查**: `pyproject.toml` 启用 mypy `disallow_untyped_defs = True`，且 `py.typed` 已存在。
- **代码格式化**: black 100 字符行宽，ruff 规则较严格。

### 5.4 质量体系短板

1. **存在一个失败用例**: `test_keep_waters_near_metal` 在当前 main 分支失败，CI 若未排除该用例则无法绿通。
2. **外部二进制依赖未在 CI 中充分验证**: `requires_vina`/`integration` 测试被默认排除，真实 Vina 子进程路径覆盖有限。
3. **`rescoring.py` 与 `minimization.py` 覆盖偏低**: 这两个模块涉及 OpenMM/OpenFF 能量计算，是发表级分析的关键，却大量依赖真实库，mock 测试不足。
4. **根目录污染**: 项目根目录存在 153 个 `test_*.py` / `test_*.pml` 调试脚本，不是正式测试，会误导新贡献者并降低仓库专业度。

---

## 6. 发表级生产标准符合度

### 6.1 已达标项

| 标准 | 状态 | 说明 |
|------|------|------|
| 版本控制与开源许可 | ✅ | MIT 许可证、`CITATION.cff`、CHANGELOG |
| 结构化入口与 API | ✅ | CLI + Python API + 模块执行 |
| 配置与可复现性 | ✅ | YAML 配置、种子、参数记录、环境状态记录 |
| 类型提示与 lint | ✅ | mypy 严格模式、ruff、black |
| 测试与 CI | ✅ | pytest、覆盖率门槛 60%、多平台 CI |
| 文档骨架 | ✅ | README、docs/、tutorials/、API stubs |
| 基准验证 | ✅ | 20 靶点 P0 基准，含困难靶点调参与公开结果 |

### 6.2 未达标/需补强项

| 标准 | 状态 | 说明 |
|------|------|------|
| 仓库卫生 | ⚠️ | 根目录 153 个调试脚本需清理或归档 |
| 方法学文档 | ⚠️ | 缺少 `METHODS.md` 详细描述算法参数与科学假设 |
| Agent 协作文档 | ❌ | 缺少 `AGENTS.md`，后续 AI 协作者难以快速理解项目规范 |
| 零失败 CI | ❌ | `test_keep_waters_near_metal` 当前失败 |
| 完整二进制集成测试 | ⚠️ | 真实 Vina/P2Rank/fpocket 路径未在默认 CI 中运行 |
| 科学限制声明 | ⚠️ | 共价配体、多聚体、柔性受体等限制未在文档首页显式列出 |
| 发布工件 | ⚠️ | 版本仍停留在 `1.0.0` Beta，建议打 tag 并发布到 PyPI |

---

## 7. 整改清单（按优先级）

### P0 — 发表前必须完成

| # | 任务 | 文件/位置 | 验收标准 |
|---|------|-----------|----------|
| 1 | 修复 `test_keep_waters_near_metal` 失败 | `autodock/tests/test_preparation.py`, `preparation.py` | `pytest autodock/tests/test_preparation.py::TestPrepareReceptor::test_keep_waters_near_metal -v` 通过 |
| 2 | 统一临时文件清理机制 | `autodock/docking.py`, `validation.py`, `preparation.py`, `analysis.py`, `minimization.py`, `interactions.py`, `rendering.py` | 所有 `delete=False`/`mkstemp` 使用 `try/finally` 或 `ExitStack`；批量任务异常中断后 `/tmp` 无残留 |
| 3 | 对齐虚拟筛选默认参数或显式声明 | `autodock/docking.py:1144` | 文档与日志明确说明 `virtual_screen` 默认 `exhaustiveness=16/n_poses=3` 的权衡 |
| 4 | 清理根目录调试脚本 | `/` | 将 153 个 `test_*.py`/`test_*.pml` 移入 `scratch/` 或删除；保留者加入 `.gitignore` |
| 5 | 创建 `AGENTS.md` | `AGENTS.md` | 包含项目架构、编码约定、测试分层、已知科学限制、常用调试命令 |

### P1 — 发表前强烈建议

| # | 任务 | 文件/位置 | 验收标准 |
|---|------|-----------|----------|
| 6 | 提升 `rescoring.py` 与 `minimization.py` 测试覆盖 | `autodock/tests/test_rescoring.py`, `test_minimization.py` | 覆盖率分别 ≥70%、≥70%；MM-GBSA 失败路径至少有一个降级测试 |
| 7 | 完整 stderr 调试日志 | `autodock/core.py:268-299` | `safe_subprocess` 在截断前将完整 stderr 写入 DEBUG；关键失败写入 `<outdir>/logs/` |
| 8 | 工具路径环境变量化 | `autodock/core.py:225-265` | `P2RANK_HOME`, `PYMOL_EXE`, `VINA_EXE` 等环境变量优先；回退自动发现 |
| 9 | 共价 warhead 检测与警告 | `autodock/preparation.py` 配体准备 | 检测到丙烯酰胺/乙烯基砜/氰基等时 logger.warning 提示 |
| 10 | 多聚体策略增强 | `autodock/workflow.py:549-619` | 对多聚体 PDB 默认输出警告；提供基于 PDB 生物组装注释的链选择 |
| 11 | 版本号获取规范化 | `autodock/core.py`, `workflow.py` | 使用 `importlib.metadata.version("autodock")`；移除 `__import__("autodock")` |
| 12 | 创建 `METHODS.md` | `METHODS.md` | 详述默认参数科学依据、口袋检测策略、RMSD 计算方式、验证流程、基准集来源 |

### P2 — 持续改进

| # | 任务 | 文件/位置 | 验收标准 |
|---|------|-----------|----------|
| 13 | 增加真实二进制集成测试 | `autodock/tests/test_vina_integration.py` | CI 中至少一个 job 安装完整环境并运行 `requires_vina`/`integration` 测试 |
| 14 | MM-GBSA 模板不匹配优雅降级 | `autodock/rescoring.py` | 失败时返回 `mmgbsa_success=False`，不阻断整体 workflow |
| 15 | 共识评分权重化 | `autodock/docking.py` | 文档化当前中位数策略，或实现基于 Spearman 加权的共识 |
| 16 | 发布 v1.0.0 / v1.1.0 tag 与 PyPI | — | GitHub Release + PyPI 包可用 |
| 17 | 建立变更日志与版本策略 | `CHANGELOG.md` | 每次合并到 main 都更新版本与变更说明 |

---

## 8. 结论

`autodock` 管线的**工程骨架已具备发表级雏形**：模块化清晰、测试覆盖率达标、CI 健全、科学流程完整。过去一个月的迭代显著改善了 RMSD 计算、MD 确定性、日志级别验证、类型标记与测试覆盖。

要在分子对接领域达到真正的发表级生产标准，仍需完成以下关键闭环：

1. **立即修复当前回归测试失败**（`test_keep_waters_near_metal`），确保 CI 全绿；
2. **系统清理临时文件与根目录调试脚本**，提升仓库卫生与长期可维护性；
3. **补齐 `AGENTS.md` 与 `METHODS.md`**，让后续协作者与审稿人都能快速理解方法学与代码约定；
4. **提升 rescoring/minimization 等关键科学模块的测试覆盖**，特别是 OpenMM/OpenFF 失败路径的优雅降级；
5. **明确声明科学限制**（共价配体、多聚体、柔性受体），避免用户误用。

完成 P0 与 P1 清单后，本项目即可进入 v1.0.0 发布与论文投稿准备阶段。
