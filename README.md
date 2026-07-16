# AI4Translate
一个基于AI大模型（deepseek v4-pro）的开源强大的英文->中文的科研论文翻译软件。

## 功能特点

- **学术论文翻译**：使用 DeepSeek LLM 将英文论文翻译为高质量中文
- **论文概要生成**：自动在翻译后的 PDF 开头生成结构化概要，包括：
  - 📌 论文标题翻译
  - 🔬 核心方法概述
  - 📊 主要发现总结
  - 🎯 结论与意义
- **AI辅读功能**：可与deepseek在终端探讨论文相关内容，保存讨论内容
- **双语对照输出**：生成中英双语对照版本，方便校对
- **保留排版**：保留原 PDF 中的图表、公式、表格
- **术语一致性**：通过专用提示词确保全文学术术语翻译一致

## 安装

```bash
pip install -r requirements.txt
```

依赖项：
- `PyMuPDF` — PDF 解析、文字擦除、重建
- `openai` — DeepSeek API 调用（OpenAI 兼容接口）
- `tqdm` — 进度条

## 快速开始

```bash
# 翻译单篇论文
python main.py paper.pdf

# 翻译 + 交互式问答
python main.py paper.pdf --chat

# 批量翻译文件夹中所有 PDF
python main.py --batch ./papers/

# 批量翻译 + 统一问答
python main.py --batch ./papers/ --chat --chat-dir ./discussions/
```

## 完整参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入 PDF 文件路径（`--batch` 时忽略） | — |
| `--batch FOLDER` | 翻译指定文件夹内所有 `.pdf` 文件 | — |
| `-o, --output` | 输出目录 | `./output/` |
| `--mode` | 输出模式：`mono` / `dual` / `both` | `mono` |
| `--chat` | 翻译后进入交互式问答聊天 | 否 |
| `--no-chat` | 跳过聊天（覆盖 `--chat`） | 否 |
| `--chat-dir` | 聊天记录保存目录 | `./chats/` |
| `--no-summary` | 不生成论文概要页 | 否 |
| `--start-page` | 起始页码（从 1 开始） | `1` |
| `--end-page` | 结束页码（0 = 最后一页） | `0` |
| `-v, --verbose` | 显示详细日志 | 否 |

## 工作流程

```
输入 PDF（单篇或批量）
    │
    ▼
Phase 1: 文本提取
    ├── 按自然段分组（PyMuPDF block → Paragraph）
    ├── 清理断词连字符（convolu- + tional → convolutional）
    ├── 检测公式行（标记为 preserve，不翻译不擦除）
    └── 保留每行的精确坐标（bbox）
    │
    ▼
Phase 2: DeepSeek 翻译
    ├── 公式/引用 → 占位符（<<<0>>>）→ 发送 API
    ├── 以完整段落为单位翻译（保留上下文语义）
    └── 翻译结果 → 还原占位符
    │
    ▼
Phase 3: 论文概要生成
    └── 提取全文关键内容 → DeepSeek 生成结构化概要
    │
    ▼
Phase 4: PDF 重建
    ├── 逐行擦除英文原文（text-only redaction，保留图表）
    └── 以段落为单位的 HTML block 写入中文（自然换行，无重叠）
    │
    ▼
Phase 5（可选）: 交互式问答
    └── 论文全文注入 LLM 上下文 → CLI 聊天 → 记录保存为 .md
```

## 输出模式

### Mono（单语版）
- 首页：论文概要（中文）
- 正文：翻译后的中文，图表公式保留，双栏布局不变

### Dual（双语版）
- 首页：论文概要（中文）
- 正文：原文页 + 译文页交替排列

## 聊天模式

### 可用命令

| 命令 | 作用 |
|------|------|
| `/papers` | 列出所有已加载论文 |
| `/paper <name>` | 切换到指定论文（或 `all` 查看全部） |
| `/clear` | 清除对话历史 |
| `/save` | 显示聊天记录保存路径 |
| `/help` | 显示帮助 |
| `/exit` | 退出聊天 |

### 聊天记录格式

对话自动保存为 Markdown 文件，包含会话时间、论文列表、Q&A 对：

```markdown
# Paper Q&A Session — 2026-07-16 16:27

## Papers Discussed
- **paper_a** (8203 chars, source: `E:\papers\paper_a.pdf`)
- **paper_b** (6501 chars, source: `E:\papers\paper_b.pdf`)

---

### [You] — 16:27:44
What is the main method proposed in this paper?

### [Assistant] — 16:27:44
该论文提出了一种基于 Transformer 架构的...
```

## 配置

在 `config.py` 中修改：

| 配置项 | 说明 |
|--------|------|
| `DEEPSEEK_API_KEY` | API 密钥 |
| `DEEPSEEK_BASE_URL` | API 地址 |
| `DEEPSEEK_MODEL` | 模型名称 |
| `SYSTEM_PROMPT_TRANSLATE` | 翻译系统提示词 |
| `SYSTEM_PROMPT_SUMMARIZE` | 概要生成提示词 |
| `MAX_CHARS_PER_BATCH` | 每次 API 调用的最大字符数 |

## 项目结构

```
deepseek-pdf-translator/
├── main.py              # CLI 入口（单篇/批量/聊天）
├── config.py            # API 密钥、提示词、字体设置
├── pdf_extractor.py     # PDF 文本提取（段落分组、公式检测、断词清理）
├── translator.py        # DeepSeek API（翻译 + 公式占位符保护 + 概要）
├── pdf_builder.py       # PDF 重建（text-only 擦除 + 段落级 HTML 写入）
├── chat.py              # 交互式问答（论文知识保留 + 对话记录保存）
├── requirements.txt     # Python 依赖
└── README.md            # 说明文档
```

## 设计参考

本项目参考了 [zotero-pdf2zh](https://github.com/guaguastandup/zotero-pdf2zh) 的整体设计思路：

- PyMuPDF 用于 PDF 的文本提取和重建
- 文本块的边界框（bounding box）定位与重绘
- 双语输出的交替页面布局
- 学术术语的一致性和准确性保障

相比 zotero-pdf2zh，本工具的改进：
- ✅ 独立运行，无需 Zotero 插件或 Flask 服务器
- ✅ 具备交互式问答功能
- ✅ 直接调用 DeepSeek API，无需中间翻译引擎
- ✅ 自动生成论文概要（核心方法 + 结论）
- ✅ 更简洁的命令行接口
