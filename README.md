# Organoid RAG Project
## Introduction

本地化 RAG 系统，基于 9,759 篇 PubMed organoid 研究文献（7,562 篇仅PMID摘要 ，2,144 篇全文 PMC），LLM为本地Qwen3.5B-Q8大语言模型，实现本地网页端和CLI端文献检索与问答。

## 技术栈

- **Embedding**: `BAAI/bge-m3` (多语言, 1024维, GPU 加速)
- **向量数据库**: ChromaDB (PersistentClient, cosine 距离)
- **重排序**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: LM Studio 本地运行 Qwen3.5-4B-Q8 (`http://localhost:1234/v1`, OpenAI 兼容 API)
- **LLM 模型路径**: 需在LM Studio自行下载并挂载 `Qwen3.5-4B-Q8_0.gguf` 或其他LLM
- **界面**: CLI (rich) + Gradio Web UI

## 项目结构

```
RAG/
├── md/                          # 原始数据（只读）
│   ├── pmids.txt                # ~10,000 PubMed IDs
│   ├── organoid_metadata.json   # 元数据 (23 MB, 9,999条)
│   ├── organoid_metadata.csv    # 同上 CSV 格式
│   └── markdown/                # 9,759 篇文章 markdown
│       ├── PMID_*.md            # 仅摘要 (~2-8 KB)
│       └── PMC*.md              # 全文 (~18 KB - 2.6 MB)
├── config.py                    # 集中配置常量
├── requirements.txt
├── src/
│   ├── cleaning.py              # 数据清洗（HTML实体、图片引用、空摘要）
│   ├── chunking.py              # 两级分块（摘要单chunk / 全文section-aware）
│   ├── embedding.py             # bge-m3 Embedding 封装
│   ├── vectorstore.py           # ChromaDB 读写
│   ├── retriever.py             # 向量搜索 + cross-encoder 重排序
│   └── generator.py             # LM Studio OpenAI API 生成
├── scripts/
│   ├── 01_clean_and_chunk.py    # 清洗分块 → data/chunks.jsonl
│   ├── 02_build_vectorstore.py  # Embedding → data/chroma_db/
│   └── 03_query.py              # CLI 交互查询
├── app.py                       # Gradio Web UI
└── data/                        # 生成产物（gitignored）
    ├── chunks.jsonl
    └── chroma_db/
```

## 数据格式

### Markdown 文件
- **PMID_\*.md**: YAML frontmatter (pmid, doi, title, authors, journal, year, keywords) + `## Abstract` + `## Keywords`
- **PMC\*.md**: 同上 frontmatter + 全文多 section（Introduction, Methods, Results, Discussion 等）
- 部分文章含中文摘要（HTML实体编码），清洗时需解码

### 元数据 JSON
字段: pmid, pmc_id, doi, title, authors, journal, year, volume, issue, pages, abstract, mesh_terms, keywords, pdf_path

## 运行流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 清洗 + 分块（~2-5 分钟）
python scripts/01_clean_and_chunk.py

# 3. Embedding + 建库（GPU ~3-5 分钟）
python scripts/02_build_vectorstore.py

# 4a. CLI 查询（需先启动 LM Studio API server）
python scripts/03_query.py

# 4b. Web UI
python app.py
```

## 关键设计决策

- **分块策略**: PMID 仅摘要文档作为单 chunk；PMC 全文按 section 拆分 (512 tokens, 64 overlap)，abstract 始终独立出一个 chunk
- **噪声过滤**: 跳过 References、Author contributions、Funding 等 section
- **元数据注入**: 每个 chunk 头部加 `[Title: ... | Journal: ... | Year: ...]`
- **检索管线**: 向量搜索 top_k=10 → cross-encoder 重排序 → 取 top_n=5 → 按文档去重
- **LLM 约束**: System prompt 要求仅基于 context 回答，引用 [PMID:XXXXX] 格式
