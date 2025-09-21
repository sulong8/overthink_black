import random
from typing import List, Dict, Tuple
from model import TargetModel, AttackerModel
import swanlab
import pathlib
from utils import read_dataset, random_sample
import json

DATA_PATH=pathlib.Path(__file__).parent / "data"

class MathProblem:
    """数学问题个体类"""
    
    def __init__(self, premises: List[str], question: str):
        self.premises = premises.copy()  # 前提列表
        self.question = question  # 问题
        self.fitness = 0.0  # 适应度分数
    
    def to_prompt(self) -> str:
        """将结构化问题转换为完整的文本提示"""
        if self.premises:
            premises_text = ", ".join(self.premises)
            return f"{premises_text}, {self.question}"
        else:
            return f"{self.question}"
    
    def copy(self):
        """创建问题的深拷贝"""
        new_problem = MathProblem(self.premises.copy(), self.question)
        new_problem.fitness = self.fitness  # 保留适应度分数
        return new_problem
    
    def __str__(self):
        return f"前提: {self.premises}, 问题: {self.question}, 适应度: {self.fitness}\n 提示: {self.to_prompt()}"

class GeneticAlgorithm:
    """遗传算法主类"""
    
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 population_size: int = 20,
                 elite_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 generations: int = 10,
                 n_samples: int = 1,
                 test_name: str = "genetic-algorithm",
                 ):

        self.model = TargetModel(model_name)
        self.population_size = population_size
        self.elite_size = int(elite_rate * population_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.n_samples = n_samples
        
        
        self.population = []
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        
        swanlab.init(
            project="overthink-black",
            name=test_name,
            config={
                "model_name": model_name,
                "population_size": population_size,
                "elite_rate": elite_rate,
                "crossover_rate": crossover_rate,
                "mutation_rate": mutation_rate,
                "generations": generations,
                "n_samples": n_samples,
            }
        )
    
    def count_tokens(self, chat_usage)->int:
        if chat_usage is None:
            return 0
        
        if hasattr(chat_usage.completion_tokens_details, "reasoning_tokens"):
            return chat_usage.completion_tokens_details.reasoning_tokens
        elif hasattr(chat_usage, "completion_tokens"):
            return chat_usage.completion_tokens
        else:
            return 0
        
        
    
    def evaluate_fitness(self) -> List:
        try:
            prompts = [ind.to_prompt() for ind in self.population]
            results = self.model.agenerate(prompts, n_samples=self.n_samples)
            if results is None:
                return []
            token_count = []
            for i, result in enumerate(results):
                _, chat_usages = result
                if chat_usages is None or len(chat_usages) == 0:
                    fitness = 0.0
                else:
                    # 计算多个样本的token数量，取平均值
                    sample_tokens = [self.count_tokens(usage) for usage in chat_usages]
                    fitness = sum(sample_tokens) / len(sample_tokens) if sample_tokens else 0.0
                self.population[i].fitness = fitness
                token_count.append(fitness)
            return token_count
        except Exception as e:
            print(f"评估适应度时出错: {e}")
            return []
    
    def evaluate_population(self):
        """评估整个种群的适应度"""
        # for individual in self.population:
        #     individual.fitness = self.evaluate_fitness(individual)
            
        #     # 更新最佳个体
        #     if individual.fitness > self.best_fitness:
        #         self.best_fitness = individual.fitness
        #         self.best_individual = individual.copy()
        fitnesses = self.evaluate_fitness()
        
        for individual in self.population:
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual.copy()
    
    def roulette_wheel_selection(self, num_parents: int) -> List[MathProblem]:
        """轮盘赌选择"""
        # 计算适应度总和
        total_fitness = sum(ind.fitness for ind in self.population)
        if total_fitness == 0:
            # 如果所有适应度都为0，随机选择
            return random.choices(self.population, k=num_parents)
        # 计算选择概率
        probabilities = [ind.fitness / total_fitness for ind in self.population]
        # 使用random.choices进行加权选择
        parents = random.choices(self.population, weights=probabilities, k=num_parents)
        return parents
        
    
    def crossover(self, parent1: MathProblem, parent2: MathProblem) -> Tuple[MathProblem, MathProblem]:
        """交叉操作：交换两个问题的前提或问题"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if random.random() < self.crossover_rate:
            # 随机选择交换方式
            if random.random() < 0.5:
                # 交换单个前提
                if parent1.premises and parent2.premises:
                    # 随机选择一个前提进行交换
                    premise1 = random.choice(parent1.premises)
                    premise2 = random.choice(parent2.premises)

                    # 替换前提
                    child1.premises = [premise2 if p == premise1 else p for p in parent1.premises]
                    child2.premises = [premise1 if p == premise2 else p for p in parent2.premises]
            else:
                # 交换问题
                child1.question, child2.question = parent2.question, parent1.question
        
        return child1, child2
    
    def mutate(self, individual: MathProblem) -> MathProblem:
        """变异操作：随机消去前提或从其他个体添加前提"""
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(["remove_premise", "add_premise"])
            
            if mutation_type == "remove_premise" and len(mutated.premises) > 0:
                # 消去前提
                mutated.premises.pop(random.randint(0, len(mutated.premises) - 1))
            
            elif mutation_type == "add_premise" and len(self.population) > 1:
                # 从其他个体添加前提
                other_individual = random.choice(self.population)
                if other_individual.premises:
                    new_premise = random.choice(other_individual.premises)
                    if new_premise not in mutated.premises:
                        insert_pos = random.randint(0, len(mutated.premises))
                        mutated.premises.insert(insert_pos, new_premise)
        
        return mutated
    
    def evolve_generation(self):
        """进化一代"""
        # 1. 适应度评估
        self.evaluate_population()
        
        # 2. 选择：精英保留
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = [ind.copy() for ind in self.population[:self.elite_size]]
        
        # 3. 生成剩余个体
        while len(new_population) < self.population_size:
            # 选择父代
            parents = self.roulette_wheel_selection(2)
            parent1, parent2 = parents[0], parents[1]
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            
            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # 添加到新种群
            new_population.extend([child1, child2])
        
        # 确保种群大小正确
        self.population = new_population[:self.population_size]
    
    def run(self, initial_problems: List[MathProblem]) -> MathProblem:
        # 初始化种群
        self.population = [problem.copy() for problem in initial_problems]
        
        print(f"开始遗传算法优化，种群大小: {self.population_size}, 代数: {self.generations}")
        
        # 进化循环
        for generation in range(self.generations):
            print(f"\n=== 第 {generation + 1} 代 ===")
            
            self.evolve_generation()
            
            # 记录当前代的最佳适应度
            current_best_fitness = max(ind.fitness for ind in self.population)
            self.fitness_history.append(current_best_fitness)
            current_mean_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            current_std_fitness = (sum((ind.fitness - current_mean_fitness) ** 2 for ind in self.population) / len(self.population)) ** 0.5
            print(f"当前代平均适应度: {current_mean_fitness}")
            print(f"当前代适应度标准差: {current_std_fitness}")
            print(f"当前代最佳适应度: {current_best_fitness}")
            print(f"历史最佳适应度: {self.best_fitness}")
            
            swanlab.log({
                "current_best_fitness": current_best_fitness,
                "best_fitness": self.best_fitness,
                "current_mean_fitness": current_mean_fitness,
                "current_std_fitness": current_std_fitness,
            })
            if self.best_individual:
                print(f"最佳个体: {self.best_individual}")
        
        # 确保返回最佳个体，如果没有则返回种群中最好的
        if self.best_individual is None:
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            return self.population[0]
        swanlab.finish()
        return self.best_individual

def question_examples() -> List[MathProblem]:
    problems = [
        MathProblem(
            ["小明有5个苹果", "小红有3个苹果"], 
            "他们一共有多少个苹果"
        ),
        MathProblem(
            ["一个正方形的边长是4cm"], 
            "这个正方形的面积是多少"
        ),
        MathProblem(
            ["一辆汽车每小时行驶60公里", "它行驶了2小时"], 
            "这辆汽车总共行驶了多少公里"
        ),
        MathProblem(
            ["一个班级有30个学生", "其中15个是男生"], 
            "女生有多少个"
        ),
        MathProblem(
            ["一本书有200页", "小李每天读20页"], 
            "小李需要多少天才能读完这本书"
        ),
        MathProblem(
            ["一个长方形的长是8厘米", "宽是5厘米"], 
            "这个长方形的周长是多少厘米"
        ),
        MathProblem(
            ["小华存了300元钱", "买了一个80元的玩具"], 
            "小华还剩多少钱"
        ),
        MathProblem(
            ["一筐鸡蛋有36个", "打碎了4个", "吃掉了8个"], 
            "筐里还剩多少个完整的鸡蛋"
        ),
        MathProblem(
            ["一袋糖果重1.5千克", "小明分给同学0.8千克"], 
            "还剩多少千克的糖果"
        ),
        MathProblem(
            ["一个圆的半径是3厘米"], 
            "这个圆的面积是多少平方厘米"
        ),
    ]
    return problems

def run(idx):
    data = read_dataset('svamp_test.json')
    initial_problems = random_sample(data, 10)
    attacker = AttackerModel("deepseek-chat")
    questions = initial_problems  # initial_problems is already a list of question strings
    
    splited_problems = attacker.split_question(questions)
    initial_problems = [MathProblem(**problem) for problem in splited_problems]
    
    print("初始问题:")
    for i, problem in enumerate(initial_problems):
        print(f"{i+1}. {problem}")

    ga = GeneticAlgorithm(
        model_name="deepseek-reasoner",
        population_size=len(initial_problems),
        elite_rate=0.4,
        crossover_rate=0.8,
        mutation_rate=0.2,
        generations=5,
        n_samples=1,
        test_name=f"svamp_{idx}"
    )
    
    best_problem = ga.run(initial_problems)
    
    print("\n" + "="*50)
    print(f"最佳问题: {best_problem}")
    print(f"最佳适应度: {ga.best_fitness}")
    print(f"最佳问题提示: {best_problem.to_prompt()}")
    return {"question": best_problem.to_prompt(), "fitness": ga.best_fitness}

if __name__ == "__main__":
    for i in range(15, 20):
        results = run(i)
        with open("svamp_result.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(results, ensure_ascii=False) + "\n")
    
