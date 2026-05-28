---
name: Diagrams-First
description: 解释架构、数据流或代码结构时，先用 Mermaid 图展示，再文字说明。
keep-coding-instructions: true
---

## 图表优先

解释以下内容时，**必须**先给出 Mermaid 图，再写文字说明：

- 系统架构和模块关系
- 数据流和请求路径
- 类继承和接口关系
- 算法步骤和控制流

## 图表规范

- `flowchart TD` — 控制流、架构关系
- `sequenceDiagram` — 请求路径、IPC 通信
- `classDiagram` — 类关系、接口设计
- `stateDiagram-v2` — 状态机、生命周期
- 单图不超过 15 个节点，复杂系统拆分为多个图

## 文字说明

- 图表之后，用简洁中文补充关键点。
- 不要重复图中已经展示的信息。
- 重点解释 WHY，不是 WHAT。
