---
name: tech-doc-translator
description: Use when translating English technical documentation (AI/ML, developer tools, API docs) into Chinese Markdown. Triggers on requests to translate docs, localize content, or convert English technical writing to Chinese.
---

# AI 技术文档翻译助手

## Overview

将英文技术文档翻译为自然、准确、专业的中文 Markdown，目标读者为中文 AI/技术从业者。

## When to Use

- 翻译英文技术文档（AI/ML、开发者工具、API 文档等）为中文
- 用户提供了待翻译的 URL 或文档内容
- 需要保留原文 Markdown 结构的本地化任务

## 翻译规则

### 术语处理

- 首次出现：「中文译名（English Term）」格式，后续直接使用中文译名
- 无公认译法的术语保留英文，必要时括号附简要解释
- 缩写词首次出现给出全称，如「大语言模型（Large Language Model, LLM）」

### 固定译法表

| English                      | 中文                        |
| ---------------------------- | --------------------------- |
| Agent                        | 智能体                      |
| Tool Use                     | 工具调用                    |
| Prompt                       | 提示词                      |
| Context Window               | 上下文窗口                  |
| MCP (Model Context Protocol) | 模型上下文协议（MCP）       |
| Permission Mode              | 权限模式                    |
| Sandbox                      | 沙箱                        |

品牌名（如 Claude Code、Hugging Face）和行业通用术语（Token、Hook）不翻译。

### 格式保留

- **Markdown 结构**：严格保持原文标题层级、段落、列表、表格
- **数学公式**：LaTeX 格式不变，行内 `$...$`，独立 `$$...$$`
- **代码与命令**：保持原样不翻译，包括代码块、行内代码、CLI 指令
- **链接**：保留原始 URL，链接文本翻译为中文
- **图片**：保留原始 URL，图片下方添加中文描述

### 网页噪音过滤

翻译时**忽略**以下网页元素：
- 页面导航栏、面包屑、侧边栏目录
- "Was this page helpful?" 等反馈组件
- "Edit this page" / GitHub 编辑链接
- 页脚版权声明、社交媒体链接
- Tab 切换控件标签（取默认 Tab 内容即可）
- 相关文章推荐、广告、CTA 横幅

### 翻译风格

- 技术文档风格：正式、准确、简洁
- 长句适当拆分，避免翻译腔，确保中文自然流畅
- 被动语态转为主动语态（如 "it is recommended" → "建议"）
- 指令性内容保持祈使句风格

## 执行流程

1. 用 `mcp__web_reader__webReader` 抓取源 URL 内容（如用户提供 URL）
2. 过滤网页噪音，提取正文
3. 按翻译规则逐段翻译
4. 输出到 `./translations/` 目录，每个页面一个文件

## 输出格式

每个文件开头包含翻译元信息：

```markdown
---
source: <原始 URL>
translated_at: <翻译日期 YYYY-MM-DD>
---

# <中文标题>

<翻译正文>
```

文件命名规则：`{序号}-{英文 slug 的简短中文描述}.md`，如 `01-how-claude-code-works.md`。

多个页面按序号逐个翻译。

## Common Mistakes

| 错误 | 正确做法 |
|------|----------|
| 翻译代码块中的变量名 | 代码和命令保持原样 |
| 保留 "it is recommended to" 翻译腔 | 转为主动语态 "建议" |
| 遗漏术语首次出现的英文标注 | 首次出现必须带英文 |
| 翻译品牌名 | 品牌名不翻译 |
| 输出网页导航/侧边栏内容 | 按噪音过滤规则忽略 |
