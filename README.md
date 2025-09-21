# Overthink Black - 遗传算法优化数学问题

这个项目实现了一个遗传算法，用于优化数学问题的表述，使其能够引发大语言模型产生更多的回复token，从而触发"过度思考"现象。

## 算法原理

### 核心概念

- **个体 (Individual)**: 一个结构化的数学问题，包含前提列表和问题描述
- **种群 (Population)**: 多个数学问题个体的集合
- **适应度函数 (Fitness Function)**: LLM对问题回复的token数量
- **遗传操作**: 交叉（交换前提/问题）和变异（消去/添加前提）

### 算法流程

1. **适应度评估**: 将每个问题发送给LLM，计算回复的token数量作为适应度分数
2. **选择**: 使用精英保留策略和轮盘赌选择法选择优秀个体
3. **交叉**: 随机交换两个问题的前提或问题部分
4. **变异**: 随机消去前提或从其他个体添加前提
5. **迭代**: 重复上述过程直到达到预设代数

## 安装和配置

### 1. 安装依赖

```bash
# 使用uv安装依赖（推荐）
uv sync

# 或使用pip
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 文件为 `.env` 并填入你的API配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
BASE_URL=https://api.openai.com/v1
API_KEY=your_openai_api_key_here
```

## 使用方法

### 基本使用

```python
from main import GeneticAlgorithm, MathProblem, create_sample_problems

# 创建示例问题
initial_problems = create_sample_problems()

# 创建遗传算法实例
ga = GeneticAlgorithm(
    model_name="gpt-3.5-turbo",
    population_size=20,
    elite_size=4,
    crossover_rate=0.8,
    mutation_rate=0.1,
    generations=10
)

# 运行算法
best_problem = ga.run(initial_problems)

print(f"最佳问题: {best_problem.to_prompt()}")
print(f"适应度分数: {best_problem.fitness}")
```

### 自定义问题

```python
# 创建自定义数学问题
custom_problems = [
    MathProblem(
        premises=["小明有10个苹果", "小红有5个苹果"],
        question="他们相差多少个苹果"
    ),
    MathProblem(
        premises=["一个圆的半径是3cm"],
        question="这个圆的周长是多少"
    )
]

# 使用自定义问题运行算法
best_problem = ga.run(custom_problems)
```

### 参数说明

- `model_name`: 使用的LLM模型名称
- `population_size`: 种群大小（建议10-50）
- `elite_size`: 精英保留数量（建议为种群大小的10-20%）
- `crossover_rate`: 交叉概率（建议0.7-0.9）
- `mutation_rate`: 变异概率（建议0.05-0.2）
- `generations`: 进化代数（建议5-20）

## 项目结构

```
overthink_black/
├── main.py              # 主程序，包含遗传算法实现
├── model.py             # LLM模型封装
├── pyproject.toml       # 项目依赖配置
├── .env.example         # 环境变量示例
├── README.md            # 项目说明
└── 算法概念设计.md       # 算法设计文档
```

## 核心类说明

### MathProblem

数学问题个体类，包含：
- `premises`: 前提列表
- `question`: 问题描述
- `fitness`: 适应度分数
- `to_prompt()`: 转换为完整提示文本

### GeneticAlgorithm

遗传算法主类，包含：
- `evaluate_fitness()`: 适应度评估
- `roulette_wheel_selection()`: 轮盘赌选择
- `crossover()`: 交叉操作
- `mutate()`: 变异操作
- `run()`: 运行算法主循环

## 运行示例

```bash
# 直接运行主程序
python main.py
```

输出示例：
```
开始遗传算法优化，种群大小: 10, 代数: 5

=== 第 1 代 ===
当前代最佳适应度: 156
历史最佳适应度: 156
最佳个体: 前提: ['小明有5个苹果', '一本书有200页'], 问题: 他们一共有多少个苹果, 适应度: 156

...

算法运行完成！
最佳问题: 已知小明有5个苹果，一本书有200页，请问他们一共有多少个苹果？
最佳适应度: 203
```

## 注意事项

1. **API费用**: 每次适应度评估都会调用LLM API，请注意控制种群大小和代数以避免过高费用
2. **模型选择**: 不同模型的token计算方式可能不同，建议使用tiktoken库进行准确计算
3. **问题质量**: 初始问题的质量会影响算法效果，建议提供多样化的初始问题
4. **参数调优**: 根据具体需求调整遗传算法参数以获得最佳效果

## 扩展功能

可以考虑添加的功能：
- 支持更多变异操作（如前提重组、问题改写等）
- 添加多目标优化（同时考虑token数量和逻辑混乱度）
- 实现并行评估以提高效率
- 添加结果可视化和分析工具

## 许可证

MIT License