# RAG 知识问答系统

基于本地 `知识库/` 目录（8 份 Markdown 文档）实现的简单 RAG（Retrieval-Augmented Generation）问答系统。

## 功能

1. **文档切分并建立向量索引**：Markdown 标题感知切分，注入标题路径；超长段落滑动窗口兜底；使用 **BGE 稠密向量 Embedding** 建索引。
2. **检索相关内容**：查询侧加 BGE query instruction，按余弦相似度返回 Top-K 片段。
3. **基于检索结果生成回答**：
   - **生成式**：配置 `OPENAI_API_KEY` 后，使用规定 Prompt 调用大模型；
   - **抽取式**：未配置 Key 时，对检索片段做要点整理并标注 `[引用N]`，避免原文整段直出。
4. **展示引用文档**：输出文件名、标题路径、相似度与摘要。
5. **无法回答时明确说明**：最高相似度 `< 0.55` 时返回「根据现有知识库，我无法回答该问题。」

## 环境要求

- Python 3.10+
- 操作系统：Windows / macOS / Linux
- 首次运行需下载 BGE 模型（约百 MB 级，需网络）

### 安装依赖

```bash
py -3 -m pip install -r requirements.txt
```

| 包 | 用途 |
|---|---|
| `numpy` | 向量运算 |
| `sentence-transformers` / `torch` | 加载 BGE Embedding |
| `openai` | 可选，生成式回答 |
| `flask` | Web 演示 |

### （可选）配置大模型（DeepSeek）

在项目根目录创建 `.env`（可参考 `.env.example`）：

```bash
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
DEFAULT_ANSWER_MODE=ai
```

网页上可切换 **AI 生成（DeepSeek）** / **抽取式整理**。命令行：

```bash
py -3 main.py --mode ai -q "企业版年度SLA是多少？"
py -3 main.py --mode extractive -q "企业版年度SLA是多少？"
```

## 在线演示（GitHub Pages）

静态网页已部署到 GitHub Pages（浏览器端检索 + 抽取式回答）：

**https://lx404616-lab.github.io/RAG-0729/**

> 说明：GitHub Pages 只能托管静态站点，无法运行 Flask/BGE。在线版使用预导出知识片段做抽取式问答；完整 BGE + DeepSeek 请本地运行 `py -3 app.py`。

若首次部署后打不开，请到仓库 **Settings → Pages → Build and deployment**，Source 选择 **GitHub Actions**，等待 Actions 成功后再访问。

### Web 网页演示（推荐）

```bash
py -3 app.py
```

浏览器打开：**http://127.0.0.1:5000**

### 命令行

```bash
py -3 main.py --rebuild          # 强制重建 BGE 索引并交互问答
py -3 main.py -q "企业版年度SLA是多少？"
py -3 main.py --demo             # 内置演示（含无法回答样例）
```

索引保存在 `vector_store/`（`embeddings.npy` + `index_meta.json`）。改模型或切分参数后请加 `--rebuild`。

## 模型及参数

配置集中在 `config.py`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | `480` | 切分块最大字符数 |
| `CHUNK_OVERLAP` | `100` | 滑动窗口重叠 |
| `CHUNK_SOFT_EXTEND` | `100` | 为凑整句允许的软扩展 |
| `TOP_K` | `4` | 检索返回片段数 |
| `SCORE_THRESHOLD` | `0.55` | BGE 余弦相似度拒答阈值 |
| `BGE_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | 稠密向量模型 |
| `BGE_QUERY_INSTRUCTION` | `为这个句子生成表示以用于检索相关文章：` | 查询前缀 |
| `BGE_BATCH_SIZE` | `32` | 编码 batch |
| `LLM_MODEL` | `gpt-4o-mini` | 生成式模型（需 API Key） |
| `LLM_TEMPERATURE` | `0.2` | 生成温度 |
| `LLM_MAX_TOKENS` | `600` | 最大生成 token |

### 向量模型说明

本实现使用 **BAAI/bge-small-zh-v1.5** 稠密向量：

- 查询侧自动加官方 instruction 前缀；
- 索引侧同时保存：内容向量、标题路径向量、事实句向量；
- 检索分 = `max(0.55·内容 + 0.45·路径, 事实句 max-pool)`，提升同义词与关键事实召回；
- 索引文件：`embeddings.npy` / `path_embeddings.npy` / `fact_embeddings.npy` / `fact_chunk_idx.npy` / `index_meta.json`。

字符级 TF-IDF 难以处理同义词与语义泛化；BGE 更适合问答召回。

若已配置 `OPENAI_API_KEY`，生成阶段走 **生成式**（规定 Prompt）；否则走 **抽取式**要点整理（标注 `[引用N]`，非原文整段直出）。

### 切分策略

1. 按 Markdown `#`～`######` 标题层级切段，维护标题栈；
2. Embedding 文本注入标题路径（重复强化）与事实要点；
3. 正文超过 `CHUNK_SIZE` 时滑动窗口兜底，优先句读断开，并允许 `CHUNK_SOFT_EXTEND` 软扩展凑整句。

### 生成式 Prompt

系统使用如下 Prompt（见 `rag/generator.py`）：

```text
你是一个严谨的知识库问答助手。请基于以下参考资料回答问题。
如果参考资料不足以回答问题，必须明确回答："根据现有知识库，我无法回答该问题。"

参考资料：
{context}

问题：{question}

要求：
1. 优先使用参考资料中的信息，不要编造
2. 引用内容请标注[引用N]
3. 若资料矛盾，请指出并说明依据
4. 回答简洁，控制在300字以内

回答：
```

## 项目结构

```text
├── 任务.txt
├── 知识库/
├── config.py
├── main.py / app.py
├── templates/index.html
├── rag/
│   ├── chunker.py     # 标题路径切分
│   ├── indexer.py     # BGE 向量索引
│   ├── generator.py   # 生成式 / 抽取式
│   └── pipeline.py
└── vector_store/      # embeddings.npy + index_meta.json
```

## 拒答策略

当检索结果为空，或最高余弦相似度 `< 0.55` 时，返回：

> 根据现有知识库，我无法回答该问题。

并标注 `[模式: 拒答(refuse) | 可回答: False]`。
