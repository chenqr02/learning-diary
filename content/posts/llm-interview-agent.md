---
date: '2026-05-09T10:00:00+08:00'
draft: false
title: '大模型面试精讲（五）：Agent'
tags: ["LLM", "面试", "Agent", "ReAct", "RAG", "MCP", "Harness", "工具调用", "多Agent", "向量检索"]
summary: '系统整理 AI Agent 高频面试题，涵盖 Agent 基础、ReAct 框架、工具调用、规划执行、多 Agent 系统、MCP 协议、Skills 与 Harness 工程。'
---

面试复习第五轮，聚焦 Agent 和 RAG。这两个方向是大模型应用落地最热门的技术，也是面试中出镜率极高的考点。

参考教材：大模型面试题精讲教材

<!--more-->

## 一、Agent 基础概念篇

### 01｜什么是 AI Agent？它的核心组件有哪些？

**AI Agent（智能体）** 是一个能够感知环境、做出决策并执行行动的自主系统，可以理解用户意图、规划任务、调用工具并完成任务。

**核心组件：**

**1. 规划模块（Planning）**

- 理解用户意图，分解复杂任务
- 制定执行计划
- 例如：将"帮我订机票"分解为"查询航班"→"选择航班"→"填写信息"→"支付"

**2. 工具调用（Tool Calling）**

- 调用外部工具和 API
- 例如：搜索、计算器、数据库查询
- 扩展模型的能力边界

**3. 记忆管理（Memory）**

- 短期记忆：当前对话的上下文
- 长期记忆：历史对话、用户偏好
- 工作记忆：当前任务的状态

**4. 反思与修正（Reflection）**

- 评估执行结果
- 发现错误并修正
- 优化执行策略

**工作流程：**

用户输入 → 理解意图 → 规划任务 → 调用工具 → 执行行动 → 评估结果 → 返回用户

**类型：**

| 类型 | 特点 |
|------|------|
| ReAct Agent | 结合推理（Reasoning）和行动（Acting），交替进行思考和行动 |
| Plan-and-Execute Agent | 先制定完整计划，再执行，适合复杂多步骤任务 |
| AutoGPT Agent | 自主执行任务，持续运行，可以自我反思和修正 |

**应用场景：** 代码生成和执行、数据分析、自动化工作流、智能助手

**挑战：** 工具调用的准确性、长期记忆管理、错误处理和恢复、安全性问题

---

### 02｜Agent 和传统的 LLM 应用有什么区别？

**传统 LLM 应用：**

- 被动响应：只能根据输入生成文本
- 无工具调用：无法使用外部工具
- 无状态管理：每次请求独立处理
- 无自主决策：需要用户明确指令

**Agent 应用：**

- 主动执行：可以自主规划并执行任务
- 工具调用：可以调用外部 API 和工具
- 状态管理：维护对话历史和任务状态
- 自主决策：可以根据情况调整策略

**核心区别：**

| 特性 | 传统 LLM | Agent |
|------|----------|-------|
| 执行方式 | 被动响应 | 主动执行 |
| 工具调用 | 不支持 | 支持 |
| 状态管理 | 无状态 | 有状态 |
| 任务规划 | 无 | 有 |
| 错误处理 | 无 | 有反思机制 |
| 应用场景 | 文本生成、问答 | 自动化任务、智能助手 |

**为什么需要 Agent？**

- 大模型无法直接访问外部信息
- 需要调用工具获取实时数据
- 复杂任务需要多步骤规划
- 需要记忆和状态管理

**示例对比：**

传统 LLM：

> 用户：今天北京天气怎么样？
> LLM：我无法获取实时天气信息，建议您查看天气应用。

Agent：

> 用户：今天北京天气怎么样？
> Agent：调用天气 API → 获取数据 → 返回：北京今天晴天，25度

---

### 03｜Agent 的记忆管理有哪些类型？各有什么特点？

**记忆类型：**

**1. 短期记忆（Short-term Memory）**

- 容量有限（通常几千 tokens）
- 快速访问，对话结束后清除
- 作用：存储当前对话的上下文
- 实现：对话历史缓冲区
- 应用：保持对话连贯性

**2. 长期记忆（Long-term Memory）**

- 容量大，持久化存储
- 需要检索机制
- 作用：存储历史对话、用户偏好、知识库
- 实现：向量数据库、关系数据库
- 应用：个性化服务、知识问答

**3. 工作记忆（Working Memory）**

- 任务相关，临时存储
- 任务完成后清除
- 作用：存储当前任务的状态和中间结果
- 实现：任务状态管理
- 应用：多步骤任务执行

**记忆管理策略：**

| 策略 | 说明 |
|------|------|
| 对话历史压缩 | 用摘要压缩长对话，保留关键信息，减少 token 消耗 |
| 向量检索 | 将记忆向量化存储，根据相关性检索，支持语义搜索 |
| 分层记忆 | 短期记忆（当前对话）→ 中期记忆（最近对话摘要）→ 长期记忆（重要信息持久化） |

**挑战：** 记忆容量限制、检索准确性、信息更新和删除、隐私和安全

---

## 二、ReAct 框架篇

### 04｜什么是 ReAct？它的工作原理是什么？

**ReAct（Reasoning + Acting）** 是一种 Agent 框架，通过交替进行推理（Reasoning）和行动（Acting）来完成任务。

**核心思想：**

- 传统方法：先推理再行动，或先行动再推理
- ReAct：推理和行动交替进行，动态调整策略

**工作流程：**

```
1. 观察（Observation）  → 获取当前环境状态
2. 思考（Thought）      → 分析当前情况，决定下一步行动
3. 行动（Action）       → 执行具体行动
4. 观察结果             → 获取行动结果，继续思考下一步
```

**示例：**

```
用户：北京的天气怎么样？

Thought: 用户想知道北京的天气，我需要调用天气API。
Action: search_weather(city="北京")
Observation: 北京今天晴天，温度25度
Thought: 我已经获取到天气信息，可以回答用户了。
Action: 返回结果给用户
```

**优势：**

- **动态调整**：根据观察结果调整策略，不需要预先制定完整计划
- **错误恢复**：发现错误可以立即修正，通过观察-思考-行动循环优化
- **灵活性**：适应不确定环境，处理意外情况

**对比：**

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| ReAct | 推理和行动交替 | 动态环境、需要探索 |
| Plan-and-Execute | 先规划再执行 | 确定环境、复杂任务 |
| Reflex | 直接行动 | 简单任务、快速响应 |

---

### 05｜ReAct 和 Plan-and-Execute 有什么区别？各适用于什么场景？

**ReAct（Reasoning + Acting）：**

- 推理和行动交替进行，动态调整策略，边思考边行动
- 工作流程：观察 → 思考 → 行动 → 观察 → 思考 → 行动 → ...
- 优势：适应不确定环境，可以及时修正错误，灵活性高
- 劣势：可能产生冗余行动，规划不够系统，执行效率可能较低

**Plan-and-Execute（规划-执行）：**

- 先制定完整计划，再按计划执行，规划和执行分离
- 工作流程：理解任务 → 制定计划 → 执行步骤1 → 执行步骤2 → ... → 完成任务
- 优势：规划系统完整，执行效率高，可以并行执行
- 劣势：不适应环境变化，错误修正困难，需要准确的环境模型

**对比：**

| 特性 | ReAct | Plan-and-Execute |
|------|-------|-------------------|
| 规划方式 | 动态规划 | 预先规划 |
| 执行方式 | 交替推理和行动 | 按计划执行 |
| 适应性 | 高 | 低 |
| 效率 | 可能较低 | 较高 |
| 错误处理 | 容易修正 | 困难 |
| 适用环境 | 不确定 | 确定 |

**选择建议：**

- 环境不确定 → 使用 ReAct
- 任务复杂但环境确定 → 使用 Plan-and-Execute
- 可以结合使用：先规划，执行时用 ReAct 调整

---

### 06｜ReAct 中的 Thought、Action、Observation 分别是什么？它们如何协作？

**三个核心组件：**

| 组件 | 作用 | 内容 | 示例 |
|------|------|------|------|
| Thought（思考） | 分析当前情况，决定下一步行动 | 推理过程、决策依据 | "用户想知道天气，我需要调用天气API" |
| Action（行动） | 执行具体操作 | 工具调用、API 请求 | `search_weather(city="北京")` |
| Observation（观察） | 获取行动结果和环境状态 | 工具返回结果、环境反馈 | "北京今天晴天，25度" |

**协作流程：**

```
1. Thought:  分析任务，决定需要什么信息
2. Action:   执行工具调用获取信息
3. Observation: 获取工具返回结果
4. Thought:  分析结果，决定下一步
5. Action:   继续执行或完成任务
6. Observation: 获取新的结果
...（循环直到任务完成）
```

**示例：** 任务：查询北京天气并给出穿衣建议

```
Thought:      用户想知道北京天气，我需要先查询天气信息。
Action:       search_weather(city="北京")
Observation:  北京今天晴天，温度25度，湿度60%

Thought:      天气信息已获取，25度是温暖的天气，我应该给出穿衣建议。
Action:       生成回答
Observation:  任务完成

最终回答：北京今天晴天，25度，建议穿轻薄衣物。
```

**关键点：**

- Thought 的质量决定 Action 的质量
- Observation 影响后续 Thought，根据观察结果调整策略
- 循环直到任务完成

**优化策略：** 限制循环次数避免死循环、验证 Observation 的可靠性、优化 Thought 的推理质量、处理 Action 失败的情况

---

## 三、工具调用篇

### 07｜Agent 如何调用工具？工具调用的流程是什么？

**工具调用流程：**

```
1. 工具定义   → 定义工具名称、描述、参数，注册到 Agent 的工具库
2. 工具选择   → Agent 根据任务选择合适的工具，使用 LLM 判断需要调用哪个工具
3. 参数生成   → LLM 根据用户输入生成工具参数，验证参数格式和有效性
4. 工具执行   → 调用工具函数或 API，获取执行结果
5. 结果处理   → 解析工具返回结果，验证结果有效性，传递给 LLM 进行后续处理
```

**示例：**

```python
# 工具定义
tools = [
    {
        "name": "search_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            }
        }
    }
]

# Agent 调用流程
# 用户输入: "北京天气怎么样？"
# 1. LLM 分析: 需要调用 search_weather 工具
# 2. 生成参数: {"city": "北京"}
# 3. 执行工具: search_weather(city="北京")
# 4. 获取结果: "北京今天晴天，25度"
# 5. LLM 生成回答: "根据查询，北京今天晴天，温度25度，适合出行。"
```

**工具调用方式：**

| 方式 | 说明 |
|------|------|
| Function Calling | LLM 输出函数调用格式，解析并执行函数，返回结果给 LLM |
| Tool Use | 使用专门的工具调用格式，支持多工具并行调用，更灵活的参数处理 |
| API 调用 | 直接调用外部 API，处理认证和错误，支持异步调用 |

**挑战：** 工具选择的准确性、参数生成的正确性、错误处理和重试、工具返回结果的验证

---

### 08｜如何设计 Agent 的工具系统？需要考虑哪些因素？

**设计原则：**

**1. 工具分类**

- 信息获取：搜索、查询、读取
- 信息处理：计算、转换、分析
- 信息存储：写入、更新、删除
- 系统操作：文件操作、网络请求

**2. 工具接口设计**

- 统一接口：所有工具使用相同的调用方式
- 清晰描述：工具名称、功能、参数说明
- 类型安全：参数类型定义明确
- 错误处理：统一的错误返回格式

**3. 工具注册和管理**

- 工具注册表：集中管理所有工具
- 权限控制：限制 Agent 可用的工具
- 版本管理：支持工具版本更新
- 动态加载：支持运行时添加工具

**设计考虑：**

| 因素 | 说明 |
|------|------|
| 工具描述 | 清晰的名称和描述、详细的参数说明、示例用法，帮助 LLM 正确选择 |
| 参数验证 | 类型检查、范围验证、必填参数检查，防止无效调用 |
| 错误处理 | 工具执行失败的处理、超时处理、重试机制、错误信息反馈 |
| 安全性 | 权限控制、输入验证、防止恶意调用、审计日志 |

**示例设计：**

```python
class Tool:
    def __init__(self, name, description, parameters, func):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def call(self, **kwargs):
        self.validate_parameters(kwargs)
        try:
            result = self.func(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

**最佳实践：** 工具描述要详细准确、参数验证要严格、错误处理要完善、支持工具组合使用、提供工具使用示例

---

### 09｜Agent 工具调用失败时应该如何处理？

**失败类型：**

| 类型 | 说明 |
|------|------|
| 参数错误 | 参数类型不匹配、缺少必填参数、参数值无效 |
| 工具执行失败 | API 调用失败、网络错误、服务不可用 |
| 超时 | 工具执行时间过长、网络延迟 |
| 权限错误 | 无权限调用工具、认证失败 |

**处理策略：**

| 策略 | 说明 |
|------|------|
| 错误信息反馈 | 返回清晰的错误信息，说明失败原因，提供修复建议 |
| 重试机制 | 对于临时性错误设置重试次数和间隔，使用指数退避策略 |
| 降级处理 | 工具失败时使用备用方案，返回部分结果，提示用户手动操作 |
| 错误恢复 | Agent 分析错误原因，尝试修正参数，选择替代工具 |

**示例：**

```python
def call_tool_with_retry(tool, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = tool.call(**params)
            if result["success"]:
                return result
            else:
                if is_retryable_error(result["error"]):
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    return result
        except TimeoutError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {"success": False, "error": "工具调用超时"}
    return {"success": False, "error": "工具调用失败，已重试多次"}
```

**Agent 层面的处理：**

- **错误分析**：Agent 分析错误信息，判断是否可以修复
- **参数修正**：根据错误信息修正参数后重试
- **替代方案**：选择其他工具或使用不同方法
- **用户通知**：告知用户失败原因和替代方案

---

## 四、规划与执行篇

### 10｜Agent 如何进行任务规划？规划算法有哪些？

**任务规划流程：**

```
1. 任务理解  → 分析用户意图，识别任务类型，提取关键信息
2. 任务分解  → 将复杂任务分解为子任务，确定依赖关系，构建任务图
3. 计划生成  → 确定执行顺序，分配资源，设置检查点
4. 计划执行  → 按顺序执行子任务，监控执行状态，处理异常情况
```

**规划算法：**

| 算法 | 说明 | 优缺点 |
|------|------|--------|
| LLM 规划 | 使用 LLM 直接生成计划 | 灵活适应性强，但可能不准确 |
| 分层任务网络（HTN） | 将任务分解为层次结构，从抽象到具体 | 适合结构化任务 |
| 状态空间搜索 | 搜索从初始状态到目标状态的路径（A*、Dijkstra） | 适合确定环境 |
| 强化学习规划 | 学习最优策略，通过试错优化 | 适合复杂环境 |

**示例：** 任务：帮我订一张从北京到上海的机票

```
规划过程：
1. 任务分解：查询航班信息 → 选择合适航班 → 填写乘客信息 → 支付订单
2. 依赖关系：查询 → 选择 → 填写 → 支付
3. 执行计划：
   Step 1: search_flights(origin="北京", dest="上海")
   Step 2: 分析结果，选择最佳航班
   Step 3: fill_passenger_info(flight_id, passenger_info)
   Step 4: pay_order(order_id)
```

**规划优化：** 并行执行（识别可并行任务）、动态调整（根据执行结果调整计划）、检查点设置（在关键步骤验证结果）

---

### 11｜Plan-and-Execute Agent 的工作流程是什么？它适合什么场景？

**工作流程：**

```
用户输入
  ↓
Planner（规划器）→ 生成执行计划：[Step1, Step2, Step3, ...]
  ↓
Executor（执行器）→ 逐步执行
  ↓
结果整合 → 返回用户
```

**示例：** 任务：分析某公司的财务数据并生成报告

```
Planner 生成计划：
1. 获取公司财务数据
2. 计算关键财务指标
3. 分析财务趋势
4. 生成可视化图表
5. 撰写分析报告

Executor 执行：
Step 1: get_financial_data(company="XXX")
Step 2: calculate_metrics(data)
Step 3: analyze_trends(metrics)
Step 4: generate_charts(data)
Step 5: write_report(analysis, charts)
```

**优势与劣势：**

| 优势 | 劣势 |
|------|------|
| 系统性强，规划完整 | 不适应环境变化 |
| 执行效率高，可并行 | 规划成本高 |
| 可解释性强，便于调试 | 难以处理意外情况 |

**适用场景：** 确定环境、复杂多步骤任务、需要可解释性的自动化工作流

**不适用场景：** 环境不确定、需要动态调整、探索性任务

---

### 12｜如何评估 Agent 的规划质量？规划失败时如何改进？

**评估指标：**

| 指标 | 说明 |
|------|------|
| 规划完整性 | 是否覆盖所有必要步骤、考虑所有依赖关系、处理边界情况 |
| 规划正确性 | 步骤顺序是否正确、依赖关系是否合理、是否能达到目标 |
| 规划效率 | 步骤数量是否最少、是否可以并行执行、执行时间是否合理 |
| 规划可执行性 | 每个步骤是否可执行、是否有必要的工具和资源、参数是否完整 |

**规划失败原因：**

- 任务理解错误：误解用户意图、遗漏关键信息
- 规划不完整：遗漏必要步骤、未考虑依赖关系
- 规划不正确：步骤顺序错误、逻辑关系错误
- 资源不足：缺少必要工具、权限不足、数据不可用

**改进策略：**

| 策略 | 说明 |
|------|------|
| 增强任务理解 | 使用更详细的提示词、多轮对话澄清需求 |
| 改进规划算法 | 使用更好的规划提示、引入规划模板、Few-shot 示例 |
| 验证和修正 | 规划后验证完整性、执行前检查可执行性、动态调整计划 |
| 错误分析 | 记录失败案例、分析失败原因、优化规划策略 |

---

## 五、多 Agent 系统篇

### 13｜什么是多 Agent 系统？它有什么优势？

**多 Agent 系统（Multi-Agent System）** 是由多个 Agent 组成的系统，每个 Agent 负责不同的任务，通过协作完成复杂目标。

**系统架构：**

- **Agent 角色分工**：规划 Agent、执行 Agent、监督 Agent、协调 Agent
- **通信机制**：消息传递、共享状态、事件驱动
- **协调策略**：集中式（有中央协调器）、分布式（Agent 自主协调）、混合式

**优势：**

| 优势 | 说明 |
|------|------|
| 专业化 | 每个 Agent 专注于特定任务，提高执行质量 |
| 并行处理 | 多个 Agent 可以并行工作，提高系统效率 |
| 可扩展性 | 容易添加新的 Agent，模块化设计 |
| 容错性 | 单个 Agent 失败不影响整体，系统更健壮 |

**示例：** 任务：开发一个 Web 应用

```
Agent 分工：
- 产品 Agent：需求分析
- 设计 Agent：UI 设计
- 前端 Agent：前端开发
- 后端 Agent：后端开发
- 测试 Agent：质量测试
- 部署 Agent：部署上线
```

**挑战：** Agent 间通信开销、协调复杂度、状态一致性、冲突解决

---

### 14｜多 Agent 系统中如何实现 Agent 间的协作和通信？

**通信方式：**

| 方式 | 说明 |
|------|------|
| 消息传递 | Agent 之间直接发送消息，支持同步和异步通信 |
| 共享状态 | 使用共享存储（数据库、内存），Agent 读写共享状态 |
| 事件驱动 | 基于事件发布订阅，Agent 订阅感兴趣的事件 |

**协作模式：**

| 模式 | 说明 |
|------|------|
| 主从模式 | 主 Agent 分配任务，从 Agent 执行，主 Agent 整合结果 |
| 对等模式 | Agent 地位平等，自主协商协作，分布式决策 |
| 层次模式 | 多层次的 Agent 结构，上层协调下层 |

**消息格式示例：**

```python
message = {
    "from": "agent_a",
    "to": "agent_b",
    "type": "task_request",
    "content": {
        "task": "analyze_data",
        "data": {...},
        "deadline": "2024-01-01"
    },
    "timestamp": "2024-01-01T10:00:00"
}
```

**协调机制：** 任务分配（根据能力分配、负载均衡）、冲突检测与协商、状态同步与一致性保证

---

### 15｜如何设计一个高效的多 Agent 系统？需要考虑哪些因素？

**设计原则：**

- **模块化设计**：每个 Agent 职责单一，接口清晰，易于替换和扩展
- **松耦合**：Agent 之间依赖最小，通过接口通信，独立部署
- **可扩展性**：容易添加新 Agent，支持动态配置

**关键因素：**

| 因素 | 说明 |
|------|------|
| Agent 角色设计 | 明确每个 Agent 的职责，避免职责重叠 |
| 通信机制 | 选择合适的通信方式，设计通信协议，处理通信失败 |
| 协调策略 | 集中式 vs 分布式，同步 vs 异步 |
| 状态管理 | 共享状态 vs 独立状态，状态同步机制 |
| 错误处理 | Agent 失败处理、任务重分配、系统恢复 |

**架构设计：**

```
┌─────────────────┐
│   协调层         │
│  (Coordinator)   │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Agent1 │ │Agent2 │
└───┬───┘ └───┬───┘
    │         │
┌───▼─────────▼───┐
│   共享状态层     │
│  (Shared State)  │
└─────────────────┘
```

**性能优化：** 并行执行、缓存中间结果、负载均衡

**监控和调试：** 日志系统（记录 Agent 行为、追踪消息传递）、可视化（显示通信关系、监控系统健康）、错误追踪

---

## 六、RAG 检索增强生成篇

### 16｜什么是 RAG？它的核心流程是什么？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种将外部知识检索与大模型生成能力结合的技术框架，通过检索相关文档来增强模型的回答质量。

**核心动机：**

- 大模型知识有截止日期，无法获取最新信息
- 大模型可能产生幻觉，生成不准确的内容
- 企业私有数据无法被大模型直接访问
- 模型参数化知识难以更新和维护

**核心流程：**

```
用户查询
  ↓
Query 处理（可选：查询改写、扩展）
  ↓
检索（Retrieval）→ 从知识库中检索相关文档
  ↓
增强（Augmentation）→ 将检索到的文档与查询拼接
  ↓
生成（Generation）→ LLM 基于增强后的上下文生成回答
  ↓
返回结果
```

**与传统 LLM 的对比：**

| 特性 | 传统 LLM | RAG |
|------|----------|-----|
| 知识来源 | 参数化知识 | 外部知识库 |
| 知识更新 | 需要重新训练 | 更新知识库即可 |
| 可追溯性 | 无法追溯 | 可以引用来源 |
| 幻觉问题 | 较严重 | 显著减少 |
| 领域适应 | 需要微调 | 更换知识库即可 |

---

### 17｜RAG 系统有哪些关键组件？如何优化每个环节？

**关键组件：**

**1. 文档处理（Document Processing）**

- 文档解析：PDF、Word、HTML 等格式转换为纯文本
- 文档分块（Chunking）：将长文档切分为合适大小的片段
- 元数据提取：保留文档来源、时间、标题等信息

**2. 向量化（Embedding）**

- 选择合适的 Embedding 模型（如 text-embedding-ada-002、BGE、M3E）
- 将文档片段转换为向量表示
- 建立向量索引

**3. 向量存储（Vector Store）**

- 向量数据库：FAISS、Milvus、Pinecone、Weaviate、Chroma
- 支持高效的相似度检索
- 支持元数据过滤

**4. 检索（Retrieval）**

- 稠密检索：基于向量相似度
- 稀疏检索：基于 BM25 等传统方法
- 混合检索：结合稠密和稀疏检索

**5. 生成（Generation）**

- Prompt 工程：设计合适的提示词模板
- 上下文拼接：将检索结果与查询组合
- LLM 生成最终回答

**各环节优化策略：**

| 环节 | 优化方法 |
|------|----------|
| 分块策略 | 固定大小分块、语义分块、递归分块；重叠窗口避免信息断裂 |
| Embedding | 选择适合目标语言/领域的模型；微调提升领域效果 |
| 检索优化 | 混合检索（稠密+稀疏）；重排序（Rerank）；查询改写 |
| 生成优化 | 精心设计 Prompt 模板；限制上下文长度；添加来源引用要求 |

---

### 18｜文档分块策略有哪些？各有什么优缺点？

**常见分块策略：**

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| 固定大小分块 | 按字符数或 token 数固定切分 | 实现简单、块大小一致 | 可能切断语义 |
| 基于分隔符分块 | 按段落、句子、标题等自然边界切分 | 保持语义完整性 | 块大小不均匀 |
| 递归分块 | 先按大分隔符切分，超长再按小分隔符切分 | 兼顾语义和大小 | 实现较复杂 |
| 语义分块 | 使用 Embedding 判断语义边界 | 语义最连贯 | 计算开销大 |
| 基于文档结构 | 按 Markdown 标题、HTML 标签等结构切分 | 保留文档结构 | 依赖文档格式 |

**分块大小的选择：**

- 太小：丢失上下文，检索到的信息不完整
- 太大：包含过多无关信息，稀释关键内容
- 一般建议：256-1024 tokens，根据具体场景调整

**重叠窗口（Overlap）：**

- 相邻块之间保留一部分重叠内容
- 避免关键信息恰好在边界处丢失
- 一般重叠 10%-20%

**最佳实践：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # 块大小
    chunk_overlap=50,      # 重叠大小
    separators=["\n\n", "\n", "。", "，", " "]  # 分隔符优先级
)
chunks = splitter.split_text(document)
```

---

### 19｜RAG 中的检索如何优化？什么是混合检索和 Rerank？

**检索优化方法：**

**1. 稠密检索（Dense Retrieval）**

- 使用 Embedding 模型将查询和文档转换为向量
- 计算向量相似度（余弦相似度、内积）
- 优点：语义理解能力强
- 缺点：对精确关键词匹配较弱

**2. 稀疏检索（Sparse Retrieval）**

- 使用 BM25、TF-IDF 等传统方法
- 基于词频和逆文档频率
- 优点：精确关键词匹配效果好
- 缺点：无法理解语义

**3. 混合检索（Hybrid Retrieval）**

- 结合稠密检索和稀疏检索
- 使用 RRF（Reciprocal Rank Fusion）等方法融合结果
- 兼顾语义理解和精确匹配

```python
# 混合检索示例
def hybrid_search(query, dense_weight=0.7, sparse_weight=0.3):
    dense_results = dense_search(query)    # 向量检索
    sparse_results = sparse_search(query)  # BM25 检索
    # RRF 融合
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        weights=[dense_weight, sparse_weight]
    )
    return fused
```

**4. 重排序（Rerank）**

- 使用 Cross-Encoder 对检索结果重新排序
- 比 Embedding 模型更精准，但更慢
- 一般用于对初检结果进行精排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, doc) for doc in retrieved_docs]
scores = reranker.predict(pairs)
reranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
```

**5. 查询改写（Query Rewriting）**

- 使用 LLM 改写用户查询，使其更适合检索
- HyDE（Hypothetical Document Embeddings）：先让 LLM 生成假设性答案，用答案去检索
- 多查询：将一个查询扩展为多个相关查询

---

### 20｜RAG 系统如何评估？有哪些评估指标？

**评估维度：**

**1. 检索质量评估**

| 指标 | 说明 |
|------|------|
| Recall@K | 前 K 个检索结果中包含正确文档的比例 |
| Precision@K | 前 K 个检索结果中相关文档的比例 |
| MRR（Mean Reciprocal Rank） | 第一个正确结果的排名倒数的平均值 |
| nDCG | 考虑排名位置的增益累积 |

**2. 生成质量评估**

| 指标 | 说明 |
|------|------|
| Faithfulness（忠实度） | 回答是否基于检索到的文档，而非模型幻觉 |
| Answer Relevance（答案相关性） | 回答是否与问题相关 |
| Context Relevance（上下文相关性） | 检索到的文档是否与问题相关 |

**3. 端到端评估**

| 指标 | 说明 |
|------|------|
| Answer Correctness | 最终回答是否正确 |
| 回答完整性 | 是否覆盖了问题的所有方面 |

**评估框架：**

- **RAGAS**：专门用于 RAG 系统评估的框架
- **TruLens**：提供 RAG 评估和追踪
- **LangSmith**：LangChain 的评估和调试平台

**评估方法：**

```python
# RAGAS 评估示例
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
print(result)
```

**最佳实践：**

- 构建高质量的评估数据集（问题 + 标准答案 + 相关文档）
- 定期评估，持续优化
- 关注检索和生成两个环节的指标
- 结合人工评估和自动评估

---

## 七、MCP 协议篇

### 21｜什么是 MCP（Model Context Protocol）？它解决了什么问题？

**MCP（Model Context Protocol）** 是 Anthropic 提出的一个开放标准协议，用于规范 LLM 应用与外部数据源、工具之间的通信方式。可以把它理解为 AI 应用世界的"USB-C 接口"——一个统一的连接标准。

**解决的核心问题：**

在 MCP 出现之前，每个 AI 应用都需要为每个外部工具/数据源单独编写集成代码，导致 $M \times N$ 的集成复杂度（M 个应用 × N 个工具）。MCP 将其简化为 $M + N$：应用实现 MCP 客户端，工具实现 MCP 服务端，通过标准协议互通。

**MCP 架构：**

```
┌─────────────────────┐
│   MCP Host（宿主）    │  ← 如 Claude Desktop、IDE、AI 应用
│  ┌────────────────┐  │
│  │  MCP Client    │  │  ← 维护与 Server 的连接
│  └───────┬────────┘  │
└──────────┼───────────┘
           │  标准协议（JSON-RPC 2.0）
    ┌──────┼──────┐
    │      │      │
┌───▼─┐ ┌──▼──┐ ┌▼────┐
│Server│ │Server│ │Server│  ← 各类工具/数据源的 MCP 服务
│  A   │ │  B   │ │  C   │
└─────┘ └─────┘ └─────┘
```

**三大核心能力：**

| 能力 | 说明 | 示例 |
|------|------|------|
| Resources | 向 LLM 暴露数据，类似 GET 请求 | 读取文件、数据库记录、API 响应 |
| Tools | 让 LLM 调用可执行的操作，类似 POST 请求 | 发送邮件、写数据库、调用外部 API |
| Prompts | 预定义的提示词模板 | 代码审查模板、数据分析模板 |

**与 Function Calling 的区别：**

| 特性 | Function Calling | MCP |
|------|-----------------|-----|
| 标准化 | 各厂商各自实现 | 统一开放协议 |
| 可发现性 | 需硬编码 | 运行时动态发现工具 |
| 跨平台 | 绑定特定模型 | 任何 LLM 宿主均可接入 |
| 生态 | 碎片化 | 可复用的服务端生态 |

---

### 22｜MCP 的通信机制是什么？有哪些传输方式？

**协议基础：** MCP 基于 JSON-RPC 2.0 协议进行通信，支持请求-响应和通知两种模式。

**生命周期：**

```
1. 初始化（Initialize）
   Client → Server: 协商协议版本和能力
   Server → Client: 返回支持的能力

2. 正常通信
   Client ↔ Server: 请求-响应、通知

3. 关闭（Shutdown）
   Client → Server: 断开连接
```

**传输方式：**

| 传输方式 | 说明 | 适用场景 |
|----------|------|----------|
| stdio | 通过标准输入/输出通信 | 本地进程、CLI 工具 |
| HTTP + SSE | HTTP 请求 + Server-Sent Events 推送 | 远程服务、Web 应用 |
| Streamable HTTP | HTTP 流式传输，支持无状态和有状态模式 | 新一代远程传输，替代 SSE |

**stdio 示例：**

```json
// Client 发送请求
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}}

// Server 返回结果
{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "file content"}]}}
```

**本地 vs 远程 MCP：**

| 特性 | 本地 MCP（stdio） | 远程 MCP（HTTP） |
|------|-------------------|-----------------|
| 部署 | 与宿主同机 | 独立部署 |
| 安全 | 操作系统级隔离 | 需要 OAuth 2.0 认证 |
| 延迟 | 低 | 有网络开销 |
| 扩展性 | 受限于单机 | 可水平扩展 |

---

### 23｜如何实现一个 MCP Server？需要考虑哪些设计问题？

**实现步骤：**

```python
# Python MCP Server 示例
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-mcp-server")

# 定义工具列表
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_database",
            description="搜索数据库中的记录",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "limit": {"type": "integer", "description": "返回数量", "default": 10}
                },
                "required": ["query"]
            }
        )
    ]

# 实现工具调用
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_database":
        results = await search_db(arguments["query"], arguments.get("limit", 10))
        return [TextContent(type="text", text=str(results))]

# 启动服务
async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read, write):
        await server.run(read, write)
```

**关键设计问题：**

| 问题 | 说明 |
|------|------|
| 工具粒度 | 太粗放则 LLM 难以精准调用，太细碎则调用次数多、上下文膨胀 |
| 描述质量 | 工具描述直接影响 LLM 的选择准确性，需要清晰、无歧义 |
| 参数校验 | 严格校验输入参数，防止非法调用 |
| 错误处理 | 返回结构化错误信息，方便 LLM 理解和重试 |
| 安全性 | 最小权限原则、输入消毒、速率限制 |
| 幂等性 | 工具调用应尽量幂等，支持安全重试 |

**安全最佳实践：**

- 敏感操作需要人类确认（human-in-the-loop）
- 实施速率限制防止滥用
- 输入验证防止注入攻击
- 日志审计所有工具调用
- 使用 OAuth 2.0 进行远程认证

---

## 八、Skills 与 Prompt 工程篇

### 24｜什么是 Agent 的 Skills？如何设计 Skill 系统？

**Skill（技能）** 是 Agent 的可复用能力单元，封装了特定任务的提示词、工具组合和执行逻辑。可以理解为 Agent 的"技能包"。

**Skill 的组成：**

```
┌─────────────────────┐
│       Skill          │
├─────────────────────┤
│ 触发条件（Trigger）   │  ← 何时使用该 Skill
│ 系统提示（Prompt）    │  ← 指导 LLM 如何执行
│ 工具集（Tools）       │  ← 该 Skill 可用的工具
│ 执行逻辑（Logic）     │  ← 预处理、后处理流程
│ 输出格式（Output）    │  ← 结果的结构化格式
└─────────────────────┘
```

**设计模式：**

| 模式 | 说明 | 示例 |
|------|------|------|
| 单一职责 | 每个 Skill 只做一件事 | 代码审查、文档生成、数据分析 |
| 可组合 | Skill 之间可以组合调用 | 先"搜索"再"总结" |
| 可覆盖 | 支持自定义覆盖默认行为 | 自定义代码风格 |
| 上下文感知 | 根据上下文调整行为 | 根据项目语言选择 lint 规则 |

**Skill 系统设计示例：**

```python
class Skill:
    def __init__(self, name, description, trigger, prompt_template, tools):
        self.name = name
        self.description = description
        self.trigger = trigger           # 触发条件
        self.prompt_template = prompt_template  # 提示词模板
        self.tools = tools               # 可用工具列表

    def should_activate(self, context) -> bool:
        """判断是否应该激活该 Skill"""
        return self.trigger(context)

    def render_prompt(self, **kwargs) -> str:
        """渲染提示词"""
        return self.prompt_template.format(**kwargs)

class SkillRegistry:
    def __init__(self):
        self.skills = {}

    def register(self, skill):
        self.skills[skill.name] = skill

    def find_skill(self, context):
        """根据上下文找到合适的 Skill"""
        for skill in self.skills.values():
            if skill.should_activate(context):
                return skill
        return None
```

**与 MCP Tools 的关系：**

- **MCP Tools** 提供原子能力（读文件、查数据库）
- **Skills** 编排多个 Tools 完成复杂任务（代码审查 = 读文件 + 分析 + 生成报告）
- Skills 是更高层的抽象，Tools 是底层的积木

---

### 25｜如何设计高质量的 Agent Prompt？有哪些技巧？

**Prompt 设计原则：**

| 原则 | 说明 |
|------|------|
| 角色明确 | 明确定义 Agent 的角色和能力边界 |
| 指令清晰 | 使用结构化的指令，避免歧义 |
| 上下文充足 | 提供足够的背景信息和约束条件 |
| 输出格式化 | 明确期望的输出格式 |

**Prompt 结构模板：**

```
# 系统提示结构
1. 角色定义：你是一个 XX 专家...
2. 能力说明：你可以使用以下工具...
3. 行为规范：你应该 / 不应该...
4. 输出格式：请以 JSON/Markdown 格式返回...
5. 示例：以下是正确的操作示例...
```

**关键技巧：**

**1. Few-shot 示例**

```
用户查询示例：
输入：北京天气怎么样？
思考：用户想知道实时天气，需要调用天气 API
行动：search_weather(city="北京")
输出：北京今天晴天，25度
```

**2. 思维链（Chain of Thought）**

```
请按以下步骤思考：
1. 分析用户的真实意图
2. 确定需要哪些信息
3. 选择合适的工具
4. 执行并验证结果
```

**3. 约束和边界**

```
重要约束：
- 不要编造信息，如果不确定请说明
- 敏感操作前必须请求用户确认
- 每次最多调用 3 个工具
```

**4. 错误处理指引**

```
如果工具调用失败：
1. 分析错误原因
2. 尝试修正参数后重试（最多 3 次）
3. 如果仍然失败，告知用户并提供替代方案
```

---

## 九、Harness 工程篇

### 26｜什么是 Agent Harness？它的核心职责是什么？

**Agent Harness（运行框架）** 是包裹在 LLM 外面的工程层，负责管理 Agent 的完整生命周期——从接收用户输入、调度 LLM 推理、执行工具调用、到返回最终结果。它不是模型本身，而是让模型"跑起来"的基础设施。

**类比理解：**

- LLM 是"大脑"，Harness 是"身体"
- LLM 是"引擎"，Harness 是"整车"
- Harness 负责：输入处理 → 模型调度 → 工具执行 → 输出格式化 → 错误恢复

**核心职责：**

```
用户输入
  ↓
┌─────────────────────────────────────┐
│           Agent Harness             │
│                                     │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ 输入处理  │  │  上下文管理       │ │
│  └────┬─────┘  └────────┬─────────┘ │
│       ↓                 ↓           │
│  ┌──────────────────────────────┐   │
│  │      LLM 调度器              │   │
│  │  (prompt 组装 → API 调用)     │   │
│  └──────────────┬───────────────┘   │
│                 ↓                   │
│  ┌──────────────────────────────┐   │
│  │      工具执行引擎            │   │
│  │  (解析调用 → 执行 → 返回)     │   │
│  └──────────────┬───────────────┘   │
│                 ↓                   │
│  ┌──────────────────────────────┐   │
│  │      循环控制器              │   │
│  │  (ReAct 循环 / 退出判断)      │   │
│  └──────────────────────────────┘   │
└─────────────────┬───────────────────┘
                  ↓
              输出结果
```

**与框架的关系：**

| 概念 | 说明 | 代表 |
|------|------|------|
| Agent 框架 | 提供构建 Agent 的抽象和工具 | LangChain、LlamaIndex、AutoGen |
| Agent Harness | 运行时管理 Agent 的完整执行 | Claude Code、Cursor、自研系统 |
| Agent 模型 | 底层 LLM 的 Agent 能力 | Claude、GPT-4、Gemini |

---

### 27｜Harness 中的循环控制和退出策略是什么？

**Agent Loop（代理循环）** 是 Harness 的核心机制，管理 LLM 与工具之间的交互循环。

**基本循环：**

```
while True:
    1. 组装 prompt（系统提示 + 对话历史 + 工具结果）
    2. 调用 LLM
    3. 解析 LLM 响应
    4. if 响应包含工具调用:
         执行工具 → 将结果加入上下文 → 继续循环
       elif 响应是最终回答:
         返回结果 → 退出循环
       elif 超过最大轮次:
         强制退出 → 返回当前结果或错误
```

**退出策略：**

| 策略 | 说明 |
|------|------|
| 正常退出 | LLM 生成最终回答，不再调用工具 |
| 最大轮次 | 超过设定的循环次数（如 25 轮）强制退出 |
| Token 预算 | 上下文 token 数超过阈值时退出 |
| 超时退出 | 总执行时间超过限制 |
| 用户中断 | 用户主动取消执行 |
| 错误退出 | 遇到不可恢复的错误 |

**实现示例：**

```python
class AgentHarness:
    def __init__(self, llm, tools, max_turns=25, max_tokens=100000):
        self.llm = llm
        self.tools = tools
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def run(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]

        for turn in range(self.max_turns):
            # 检查 token 预算
            if self.count_tokens(messages) > self.max_tokens:
                return self.summarize_and_exit(messages)

            # 调用 LLM
            response = self.llm.chat(messages, tools=self.tools)

            # 判断是否结束
            if response.is_final_answer:
                return response.content

            # 执行工具调用
            for tool_call in response.tool_calls:
                result = self.execute_tool(tool_call)
                messages.append({"role": "tool", "content": result})

        return "达到最大轮次，任务未完成"
```

---

### 28｜Harness 中如何管理上下文窗口？有哪些压缩策略？

**上下文窗口管理** 是 Harness 工程中最关键的技术挑战之一。随着对话和工具调用的积累，上下文会快速增长并超出模型限制。

**核心问题：**

- 每次工具调用的结果都占用上下文空间
- 多轮 ReAct 循环产生大量中间状态
- 上下文溢出会导致模型无法正常工作

**压缩策略：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 滑动窗口 | 只保留最近 N 条消息 | 对话历史管理 |
| 摘要压缩 | 用 LLM 将长对话压缩为摘要 | 长对话场景 |
| 工具结果裁剪 | 对工具返回结果进行截断或摘要 | 工具返回大量数据 |
| 选择性保留 | 根据相关性保留重要信息 | 多轮对话 |
| 分层压缩 | 近期详细、远期摘要 | 长期运行的 Agent |

**实现示例：**

```python
class ContextManager:
    def __init__(self, max_tokens=100000, llm=None):
        self.max_tokens = max_tokens
        self.llm = llm

    def compress(self, messages: list) -> list:
        total = self.count_tokens(messages)
        if total <= self.max_tokens:
            return messages

        # 策略1: 裁剪工具结果
        messages = self.trim_tool_results(messages)

        # 策略2: 摘要压缩早期对话
        if self.count_tokens(messages) > self.max_tokens:
            messages = self.summarize_early_messages(messages)

        # 策略3: 滑动窗口兜底
        if self.count_tokens(messages) > self.max_tokens:
            messages = self.sliding_window(messages)

        return messages

    def summarize_early_messages(self, messages):
        """将早期消息压缩为摘要"""
        system = messages[0]  # 保留系统提示
        early = messages[1:-10]  # 取早期消息
        recent = messages[-10:]  # 保留最近消息

        summary = self.llm.chat([
            {"role": "user", "content": f"请将以下对话压缩为简洁摘要：\n{early}"}
        ])
        return [system, {"role": "system", "content": f"历史摘要：{summary}"}] + recent
```

---

### 29｜Harness 如何实现安全控制和权限管理？

**安全挑战：**

Agent 具备执行工具的能力，如果缺乏安全控制，可能导致数据泄露、越权操作、资源滥用等严重后果。

**安全控制层次：**

```
┌─────────────────────────┐
│   第一层：输入验证       │  ← 过滤恶意输入、注入攻击
├─────────────────────────┤
│   第二层：权限控制       │  ← 工具级权限、数据级权限
├─────────────────────────┤
│   第三层：执行沙箱       │  ← 隔离执行环境、限制资源
├─────────────────────────┤
│   第四层：人类确认       │  ← 敏感操作需人工审批
├─────────────────────────┤
│   第五层：审计日志       │  ← 记录所有操作、便于追溯
└─────────────────────────┘
```

**权限模型：**

| 维度 | 说明 | 示例 |
|------|------|------|
| 工具级权限 | 控制 Agent 可调用哪些工具 | 只允许读取，不允许写入 |
| 数据级权限 | 控制可访问的数据范围 | 只能访问特定目录 |
| 操作级权限 | 控制操作的类型和范围 | 禁止删除操作 |
| 速率限制 | 限制调用频率 | 每分钟最多 10 次 API 调用 |

**Human-in-the-Loop（人类确认）：**

```python
class SafetyController:
    DANGEROUS_ACTIONS = ["delete", "drop", "send_email", "pay", "deploy"]

    def require_confirmation(self, tool_call) -> bool:
        """判断是否需要人类确认"""
        # 危险操作必须确认
        if tool_call.name in self.DANGEROUS_ACTIONS:
            return True
        # 写操作需要确认
        if tool_call.name.startswith("write_"):
            return True
        return False

    def execute_with_safety(self, tool_call):
        if self.require_confirmation(tool_call):
            approved = self.ask_user_approval(tool_call)
            if not approved:
                return "操作已被用户取消"
        return self.execute_tool(tool_call)
```

**沙箱执行：**

- Docker 容器隔离代码执行
- 文件系统限制（只读挂载、目录白名单）
- 网络限制（禁止访问内网）
- 资源限制（CPU、内存、执行时间）

---

### 30｜如何设计一个生产级的 Agent Harness？需要考虑哪些工程问题？

**生产级 Harness 的关键工程问题：**

| 问题 | 说明 | 解决方案 |
|------|------|----------|
| 可靠性 | LLM 调用可能失败或超时 | 重试机制、降级策略、超时控制 |
| 可观测性 | 需要追踪 Agent 的每一步决策 | 结构化日志、链路追踪、可视化 |
| 可扩展性 | 支持多种 LLM、多种工具 | 抽象接口、插件化设计 |
| 成本控制 | LLM 调用和工具调用都有成本 | Token 预算、缓存、模型路由 |
| 测试性 | Agent 行为难以预测和测试 | 回归测试、Eval 框架、Mock |

**架构设计：**

```python
class ProductionHarness:
    def __init__(self, config):
        self.llm_router = LLMRouter(config.models)      # 多模型路由
        self.tool_registry = ToolRegistry(config.tools)   # 工具注册
        self.context_manager = ContextManager(config)     # 上下文管理
        self.safety_controller = SafetyController(config) # 安全控制
        self.logger = StructuredLogger(config.logging)    # 结构化日志
        self.cache = ResponseCache(config.cache)          # 响应缓存
        self.metrics = MetricsCollector(config.metrics)   # 指标采集

    async def run(self, user_input, session_id):
        trace_id = self.logger.start_trace(session_id)

        try:
            # 检查缓存
            cached = self.cache.get(user_input)
            if cached:
                return cached

            # 执行 Agent 循环
            result = await self.agent_loop(user_input, trace_id)

            # 缓存结果
            self.cache.set(user_input, result)
            return result

        except Exception as e:
            self.logger.error(trace_id, e)
            return self.fallback(e)

        finally:
            self.logger.end_trace(trace_id)
            self.metrics.flush()
```

**可观测性设计：**

```python
# 每次循环记录
trace_event = {
    "turn": 3,
    "llm_call": {
        "model": "claude-sonnet-4-6",
        "input_tokens": 2048,
        "output_tokens": 512,
        "latency_ms": 1200
    },
    "tool_calls": [
        {"name": "search", "args": {"query": "..."}, "result_size": 1024, "latency_ms": 300}
    ],
    "context_tokens": 15000,
    "decision": "continue"  # or "final_answer"
}
```

**成本控制策略：**

- **模型路由**：简单任务用小模型，复杂任务用大模型
- **Token 预算**：设置每轮对话的最大 token 数
- **缓存**：对相同或相似查询缓存结果
- **批量处理**：合并多个工具调用减少往返

**测试策略：**

| 测试类型 | 说明 |
|----------|------|
| 单元测试 | 测试各组件独立功能 |
| 集成测试 | 测试组件间协作 |
| 回归测试 | 用固定 case 验证行为不变 |
| Eval 评估 | 用 LLM 评估 Agent 输出质量 |
| 混沌测试 | 模拟故障验证容错能力 |

---

## 总结

本文涵盖了 AI Agent 和 RAG 的高频面试题：

| 模块 | 核心要点 |
|------|----------|
| Agent 基础 | Agent 定义、核心组件、与传统 LLM 的区别、记忆管理 |
| ReAct 框架 | 推理与行动交替、与 Plan-and-Execute 对比、TAO 协作机制 |
| 工具调用 | 调用流程、工具系统设计、失败处理策略 |
| 规划与执行 | 任务规划算法、Plan-and-Execute 流程、规划质量评估 |
| 多 Agent 系统 | 系统优势、协作通信机制、高效系统设计 |
| RAG | 核心流程、分块策略、检索优化（混合检索+Rerank）、评估指标 |
| MCP 协议 | 统一工具协议、通信机制、MCP Server 实现 |
| Skills | Skill 系统设计、Prompt 工程技巧 |
| Harness 工程 | 循环控制、上下文压缩、安全权限、生产级架构 |

**面试建议：**

- 理解 Agent 的核心概念，能画出工作流程图
- 熟悉 ReAct 和 Plan-and-Execute 的区别，能结合场景选择
- 掌握工具调用的设计和错误处理
- 了解 RAG 的完整链路和各环节优化方法
- 理解 MCP 协议的设计思想，能对比 Function Calling
- 了解 Harness 的工程挑战：上下文管理、安全控制、可观测性
- 能够结合实际场景分析问题，给出系统设计方案
