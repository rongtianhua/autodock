# 共价配体检测与多聚体受体策略 — 设计策略文档

**版本**: 1.0  
**日期**: 2026-07-01  
**范围**: P1/P3 增强（共价/反应性配体检测、多聚体受体默认策略）  
**目标**: 在修改代码前，明确科学实践、集成点、API 变更与测试方案，确保与现有 AutoDock Vina 非共价主路径最小侵入兼容。

---

## 1. 共价 / 反应性配体检测

### 1.1 设计目标

- **轻量级**: 不引入完整共价对接引擎（AutoDock4 covalent、GOLD covalent、DOCKovalent 等）的复杂度。
- **非侵入**: 不修改 Vina 打分或采样，仅在配体准备与结果层做**检测、标注与警告**。
- **科学可解释**: 基于被广泛引用的 warhead SMARTS 分类（Baillie 2016、Schirmeister 2020、Singh 2025 综述）和反应性残基兼容性矩阵。
- **可扩展**: 为后续 `--covalent` / `covalent_docking=True` 模式保留接口，但当前版本不实现几何约束或反应后拓扑。

### 1.2 背景与最佳实践

AutoDock Vina **本身不支持共价对接**。文献中处理共价配体的常见做法有三种：

1. **非共价预结合模式**（Vina / SCAR / SILCS-covalent）: 将配体当作完整 warhead 进行非共价对接，靠后续 MD 或人工判断反应几何。该方法的假设是 `k_on << k_inact`，即非共合步骤是限速步。
2. **产物态 / tethered 对接**（AutoDock4 flexible sidechain、GOLD link-atom）: 将配物与受体残基连接后作为整体采样，需要专门的共价对接软件。
3. **几何可行性门控**（PoseBusters + 反应原子距离/角度过滤）: 在非共价对接后，检查 warhead 与目标亲核原子是否处于近攻击构象（NAC）。

本管线当前以 **Vina 非共价对接** 为主引擎，因此只采用 **方案 1 + 方案 3 的检测层**，不实现方案 2 的产物态拓扑修改。

### 1.3 Warhead SMARTS 分类

| Warhead 类别 | SMARTS（示例，RDKit 可用） | 典型反应性残基 | 备注 |
|--------------|---------------------------|----------------|------|
| 丙烯酰胺 / 甲基丙烯酰胺 | `[C;$(C=C-C(=O)N)]`（Michael acceptor） | Cys | 最常见（~98% 已上市共价药） |
| 氯乙酰胺 / 溴乙酰胺 | `[C;$(C-C(=O)N)](Cl)` / Br | Cys | 高反应性，选择性较低 |
| 乙烯基砜 | `[C;$(C=C-S(=O)(=O))]` | Cys | 不可逆，较慢 |
| 磺酰氟 / 氟硫酸酯 | `S(=O)(=O)F` / `OS(=O)(=O)F` | Ser, Thr, Tyr, Lys, Cys | 新趋势（SuTEx / 酪氨酸共价） |
| 醛 | `[CX3H1](=O)` | Lys, Cys, N-terminus | 可逆 / 半缩胺 |
| 硼酸 / 硼酸酯 | `B(O)O` | Ser, Thr, Tyr | 可逆，蛋白酶体抑制剂常见 |
| 腈 | `C#N`（与亲核试剂加成） | Cys, Ser | 可逆（如 niraparib 类似物） |
| 环氧化物 | `C1OC1` | Cys, Asp, Glu | 较少见 |
| 马来酰亚胺 | `O=C1C=CC(=O)N1` | Cys | 高选择性 |
| 乙炔基 / 炔酰胺 | `C#C-C(=O)N` | Cys | 不可逆 |
| 卤代烷（非 α-羰基） | `[C][Cl,Br,I]` | Cys, Lys, Asp, Glu | 非特异性，常见于探针 |

> 注：SMARTS 需要 RDKit 支持。部分模式需通过 `Chem.MolFromSmarts` 并考虑芳香性/脂肪族版本。实际实现时优先使用 **精确子结构匹配**（`HasSubstructMatch`），避免过度宽泛导致误报。

### 1.4 反应性残基兼容性矩阵

| 亲核残基 | 侧链亲核原子 | 兼容 warhead（优先级） |
|----------|--------------|------------------------|
| Cys      | SG (thiolate) | 丙烯酰胺、马来酰亚胺、乙烯基砜、氯乙酰胺、环氧化物、腈、乙炔酰胺 |
| Ser      | OG (alkoxide) | 磺酰氟、硼酸、腈、醛 |
| Thr      | OG1           | 磺酰氟、硼酸、腈、醛 |
| Tyr      | OH (phenolate)| 磺酰氟、氟硫酸酯、醛 |
| Lys      | NZ (amine)    | 醛、磺酰氟、卤代烷 |
| Asp/Glu  | OD/OE (carboxylate) | 环氧化物、卤代烷（少见） |
| His      | NE2/ND1       | 卤代烷、醛（少见） |
| Met      | SD (thioether)| 卤代烷（甲基化） |

### 1.5 集成点设计

#### 1.5.1 新增模块: `autodock/covalent.py`

独立模块，职责单一：

```python
class WarheadMatch(NamedTuple):
    name: str
    smarts: str
    reactive_atoms: tuple[int, ...]   # 配物中参与共价键形成的原子索引
    compatible_residues: tuple[str, ...]

class CovalentAnnotation:
    has_warhead: bool
    warhead_matches: list[WarheadMatch]
    recommended_residues: set[str]
    risk_level: str  # "low" | "medium" | "high"
```

核心函数：

- `detect_covalent_warheads(smiles: str) -> CovalentAnnotation`: 对 SMILES 执行 SMARTS 匹配，返回所有命中。
- `recommend_reactive_residues(receptor_pdbqt: str, center, box_size, residue_types: set[str]) -> list[dict]`: 在受体网格内搜索指定类型的亲核残基，记录链、残基号、原子名、坐标。
- `check_geometric_feasibility(ligand_pdbqt, receptor_pdbqt, match: WarheadMatch, max_dist: float = 5.0) -> bool`: 简单距离门控（当前版本仅用于报告，不打断流程）。

#### 1.5.2 配体准备层: `preparation.py`

- 在 `prepare_ligand()`、`prepare_ligand_from_file()`、`prepare_ligand_from_sdf()` 中可选调用 `detect_covalent_warheads()`。
- 新增参数 `covalent_check: bool = False`（默认关闭，避免破坏现有 API）。
- 当检测到 warhead 时，在 `logger.warning` 中输出明确信息：
  - warhead 名称
  - 兼容的反应性残基
  - 说明 Vina 将以非共价方式处理，结果能量不能直接解释为共价结合亲和力
- 返回值保持为 `str`（PDBQT 路径），不改变现有签名；若需要传递 annotation，可通过 `output_json` 或 `metadata` 参数扩展（见 1.5.4）。

#### 1.5.3 对接结果层: `core.py`

在 `DockingResult` dataclass 中新增字段（可空，不影响现有序列化）：

```python
# Covalent annotation
is_covalent_ligand: bool | None = None
warhead_names: list[str] = field(default_factory=list)
covalent_recommendation: str | None = None  # 例如 "Potential Cys covalent binder; Vina scored non-covalently"
```

`to_dict()` 自动包含这些字段；旧数据读取时因 dataclass 默认值而兼容。

#### 1.5.4 工作流层: `workflow.py`

- 新增参数 `covalent_check: bool = False`。
- 在配体准备步骤之后、对接之前调用检测；若检测到 warhead：
  - 记录到 `DockingWorkflowResult.warnings`。
  - 在结果 JSON / CSV 中增加 `covalent_warning` 列。
- 保留现有 Vina 对接路径不变。

#### 1.5.5 CLI

- `prepare-ligand` 增加 `--covalent-check / --no-covalent-check`（默认 `--no-covalent-check`，避免改变默认行为）。
- `workflow` 增加 `--covalent-check`。
- `virtual-screen` 增加 `--covalent-check`，输出 CSV 中新增 `covalent_warning` 列。

### 1.6 风险分级

| 风险等级 | 条件 | 建议行为 |
|----------|------|----------|
| low      | 检测到 warhead，但受体网格内无兼容亲核残基 | 信息日志；结果标注 |
| medium   | 检测到 warhead，且受体网格内有兼容亲核残基 | warning；结果标注；提示用户考虑共价对接工具 |
| high     | 检测到多个 warhead 或高反应性 warhead（氯乙酰胺、磺酰氟等） | warning；结果标注；建议人工复核 |

### 1.7 回退与兼容

- 若 RDKit 未安装，`covalent_check` 自动降级为 `False` 并记录 debug 日志。
- 若 SMARTS 解析失败，捕获异常并继续准备流程，不阻塞。
- 不改变默认行为：无 `--covalent-check` 时，代码路径与现在完全一致。

### 1.8 测试策略

| 测试对象 | 内容 |
|----------|------|
| `covalent.py` 单元测试 | 对已知共价配体（osimertinib 丙烯酰胺、ibrutinib 丙烯酰胺、neratinib 丙烯酰胺、氯乙酰胺示例）检测为真；对常见非共价配体（aspirin、ibuprofen、N3）检测为假。 |
| SMARTS 覆盖 | 每个 warhead 类别至少一个正例与负例。 |
| 反应性残基推荐 | mock 受体 PDBQT，验证在网格内正确返回 Cys/Ser 等。 |
| 工作流集成 | mock 配体准备与 RDKit，验证 `covalent_check=True` 时结果 JSON 含 warning。 |
| CLI 参数 | 验证 `--covalent-check` 被正确传递且默认关闭。 |
| 降级路径 | 无 RDKit 时 `covalent_check` 不报错。 |

---

## 2. 多聚体受体策略

### 2.1 当前问题

`run_docking_workflow()` 在以下条件下触发多聚体处理：

- `receptor_source == "PDB"`
- 不对称单元链数 > 1
- `get_pdb_assembly_info()` 返回 `is_monomeric == True`

当前 `receptor_multichain_strategy="auto"` 默认映射为 `"extract_single"`，即**直接抽取单链**。这对于“晶体中多条相同序列链但生理上为单体”的情况（如 4F9Z）是正确的。然而：

- 若 PDB 组装信息被错误标注为单体（PISA/EPPIC 对高阶寡聚体错误率可达 25-45%，Dey 2021），会丢失功能位点。
- 对真实同源二聚体、异源二聚体或更高阶寡聚体，默认抽取单链会切断结合口袋，导致对接失败或伪阳性。
- 当前策略未区分 **monomeric ASU** 与 **true multimeric assembly**。

### 2.2 科学最佳实践

1. **优先信任实验证据**: PDB 的 `pdbx_struct_assembly` 元数据是作者/PISA 对生理寡聚态的注释，但仍应：
   - 检查 `oligomeric_details` 是否包含 "monomeric" / "homodimeric" / "heterodimeric" / "tetrameric" 等关键词。
   - 检查 `oligomeric_count`（单链数）。
2. **结构证据辅助**: 若元数据不可用，可通过 `Biopython` 分析不对称单元内链间界面面积（>800-1000 Å² 通常视为真实生物界面）。
3. **保守默认**: 当无法确定时，**保留多链结构** 比激进抽取单链更安全；Vina 可以处理多链受体，只是计算量稍大。
4. **用户显式控制**: 保持 `receptor_multichain_strategy` 的显式选项，让用户覆盖自动推断。

### 2.3 建议的决策树

```
if receptor_source != "PDB" or len(asymmetric_chains) <= 1:
    保持原受体
else:
    asm_info = get_pdb_assembly_info(receptor_id)
    oligomeric_count = asm_info.get("oligomeric_count", 0)
    details = asm_info.get("oligomeric_details", "").lower()

    if oligomeric_count == 1 or "monomeric" in details:
        # 单体：抽取单链（原行为）
        apply_strategy("extract_single")
    elif oligomeric_count == 2 or "dimer" in details:
        # 二聚体：默认保留多链（multichain），用户可显式要求 extract_single
        if "auto" in strategies:
            strategies = ["multichain"]
        apply_strategy(用户指定或 multichain)
    else:
        # 三聚体及以上：保留多链，并 warning 提示高阶寡聚体可能需特殊处理
        if "auto" in strategies:
            strategies = ["multichain"]
        log.warning("PDB %s 为 %s（%d 链），默认保留完整生物组装；"
                    "如需单链对接请显式设置 receptor_multichain_strategy='extract_single'",
                    receptor_id, details, oligomeric_count)
        apply_strategy(用户指定或 multichain)
```

### 2.4 策略语义调整

当前策略：`auto` → `extract_single`

建议策略：`auto` 根据寡聚态推断：

| 寡聚态 | auto 默认行为 | 说明 |
|--------|---------------|------|
| 单体 / 未知 | `extract_single` | 与当前行为一致，保持向后兼容 |
| 同源二聚体 / 异源二聚体 | `multichain` | 保留功能界面 |
| 三聚体及以上 | `multichain` + warning | 保留完整组装，提示用户 |

保留 `receptor_multichain_strategy` 接受字符串或列表，允许用户显式覆盖：

```python
run_docking_workflow(
    receptor_id="2HU4",
    receptor_multichain_strategy="extract_single",  # 强制单链
)
run_docking_workflow(
    receptor_id="1ABC",
    receptor_multichain_strategy="multichain",      # 强制多链
)
```

### 2.5 实现细节

#### 2.5.1 `workflow.py` 修改范围

修改集中在第 550-555 行的策略解析逻辑：

```python
# 旧
if "auto" in strategies:
    strategies = ["extract_single"]

# 新
if "auto" in strategies:
    if oligomeric_count == 1 or "monomeric" in details:
        strategies = ["extract_single"]
    else:
        strategies = ["multichain"]
        if oligomeric_count >= 3 or any(w in details for w in ("trimer", "tetramer", "hexamer", "octamer")):
            logger.warning(f"PDB {receptor_id}: biological assembly suggests ...")
```

其余 `multichain` / `extract_single` / `alphafold` 分支保持不变。

#### 2.5.2 `get_pdb_assembly_info()` 增强

当前函数已返回 `oligomeric_count`、`oligomeric_details`、`chains_per_assembly`。确保：

- `is_monomeric` 计算准确（`all(c == 1 for c in oligomeric_counts)`）。
- 当 RCSB API 不可用时，`oligomeric_count` 返回 `0`，触发 `auto` 回退到 `extract_single`（保持安全保守）。
- 可考虑缓存 assembly info 到 `workflow_state.json` 避免重复请求。

#### 2.5.3 文档与用户提示

- `METHODS.md` 增加“多聚体受体处理”章节，说明 `auto` 策略的推断规则。
- CLI help 更新 `receptor_multichain_strategy` 的描述。
- 日志中明确打印最终采用策略及原因。

### 2.6 回退与兼容

- 向后兼容：显式 `receptor_multichain_strategy="extract_single"` 仍强制抽取单链。
- 离线场景：RCSB API 失败时 `oligomeric_count=0`，`auto` 回退到 `extract_single`，与当前行为一致。
- 对 AlphaFold / SWISS-MODEL / 本地文件：不触发多聚体逻辑，无变化。

### 2.7 测试策略

| 测试对象 | 内容 |
|----------|------|
| `auto` 单体 | mock `oligomeric_count=1` + 多链 ASU，验证仍调用 `extract_single`。 |
| `auto` 二聚体 | mock `oligomeric_count=2` / `homodimeric`，验证默认使用 `multichain`，不抽取单链。 |
| `auto` 高阶寡聚体 | mock `oligomeric_count=4` / `tetrameric`，验证使用 `multichain` 并记录 warning。 |
| 显式覆盖 | 验证 `receptor_multichain_strategy="extract_single"` 在二聚体场景下仍抽取单链。 |
| API 失败 | mock `get_pdb_assembly_info` 抛出异常，验证流程不崩溃并回退到 `multichain` 或 `extract_single`（需在设计评审后确定）。 |
| 现有测试更新 | `test_strategy_auto_defaults_to_extract_single` 需要拆分为“单体”与“非单体”两个测试。 |

---

## 3. 与现有管线的兼容矩阵

| 改动 | 默认行为改变 | API 签名改变 | 序列化格式改变 | 向后兼容 |
|------|--------------|--------------|----------------|----------|
| 共价检测（默认关闭） | 否 | 仅新增可选参数 | `DockingResult` 新增可空字段 | ✅ 是 |
| 共价检测（CLI 显式开启） | 是（开启时） | 新增 CLI flag | CSV/JSON 新增列 | ✅ 是 |
| 多聚体 `auto` 策略 | 是（对真实二聚体+） | 无 | 无 | ✅ 显式策略仍可用 |
| `DockingResult` 新增字段 | 否 | 无 | 新增键 | ✅ 默认值兼容 |

---

## 4. 待决策事项

在编码前，请确认以下事项：

1. **共价检测默认是否开启？**
   - 建议：默认关闭，通过 `--covalent-check` 开启，避免对现有批量筛选产生日志噪音。
2. **是否将 `DockingResult` 扩展为返回 `CovalentAnnotation` 对象而非仅字符串？**
   - 建议：当前仅增加简单字段，复杂对象可在 v2 引入。
3. **多聚体 API 失败时的回退行为？**
   - 选项 A: 保守保留多链（multichain）—— 更安全，但可能改变部分离线跑的结果。
   - 选项 B: 维持当前 `extract_single` —— 完全向后兼容，但可能丢失功能位点。
   - 建议：选项 A，并在日志中说明无法获取 assembly info 故保守保留多链。
4. **是否需要引入 Biopython 接口面积计算作为第二证据？**
   - 建议：P1 阶段不引入；仅使用 RCSB 元数据。P2 可扩展。

---

## 5. 下一步行动计划

1. 评审并确认本设计策略（尤其第 4 节待决策事项）。
2. 根据确认结果，按以下顺序编码：
   - 新增 `autodock/covalent.py` + 单元测试
   - 扩展 `DockingResult` + `to_dict` 测试
   - 在 `preparation.py` / `workflow.py` / `docking.py` / `cli.py` 中接入共价检测
   - 修改 `workflow.py` 多聚体策略决策树 + 更新 `test_workflow.py`
   - 更新 `METHODS.md` 与 CLI help
3. 执行完整测试套件，确保覆盖率仍 ≥60% 且不低于当前 75.93%。
4. 补充基准验证：选择 1-2 个已知共价配体（如 SARS-CoV-2 Mpro 共价抑制剂）和 1 个真实二聚体靶点，手动验证行为符合预期。

---

## 6. 参考文献

- Baillie, T. A. (2016). Targeted covalent inhibitors for drug design. *Angew. Chem. Int. Ed.*
- Schirmeister, T., & Kesselring, J. (2020). Covalent inhibitors of cysteine proteases. *ChemMedChem*.
- Singh, N., et al. (2025). The covalent docking software landscape. *PMC12753312*.
- Dey, S., et al. (2021). PDB-wide identification of physiological hetero-oligomeric assemblies. *Nat. Commun.*
- Krissinel, E., & Henrick, K. (2007). Inference of macromolecular assemblies from crystalline state. *J. Mol. Biol.* (PISA)
- AutoDock Vina documentation: https://vina.scripps.edu/
