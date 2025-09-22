import random
import argparse
import json
import re
from typing import List, Dict, Tuple

from torch.utils.hipify.hipify_python import file_add_header

from model import TargetModel, AttackerModel, DeepSeek
import copy
import numpy as np
import swanlab
from fastchat.conversation import get_conv_template
from json import JSONDecodeError
import traceback

file_path = 'data_old/math_test.jsonl'
file_out_path = 'data_old/math_change.jsonl'
flie_result_path = 'data_old/math_result.json'
flie_all_path = 'data_old/math_all.json'
overthink_tokens = ['wait', 'Alternatively', 'but', "Wait", "Maybe", "maybe"]
attitude_tokens = ['difficult','complicated','diverse','multiple']

def read_questions_from_jsonl(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'question' in data:
                        questions.append(data['question'])
                except json.JSONDecodeError as e:
                    print(f"解析JSON行时出错: {e}")
                    continue
    return questions

class Problem:

    def __init__(self, son_questions: List[str], ):
        self.son_questions = son_questions.copy()  # 前提列表
        self.fitness = 0  # 适应度分数

    def to_prompt(self) -> str:
        """将结构化问题转换为完整的文本提示"""
        premises_text = ",".join(self.son_questions)
        return premises_text

    def copy(self):
        """创建问题的深拷贝"""
        new_problem = Problem(self.son_questions.copy())
        new_problem.fitness = self.fitness  # 保留适应度分数
        return new_problem

    def __str__(self):
        return f" 问题:{self.son_questions}, 适应度: {self.fitness}"

class GeneticAlgorithm:
    """遗传算法主类"""

    def __init__(self,dataset,num,
                 model_name: str = "gpt-3.5-turbo",
                 population_size: int = 20,
                 elite_rate: int = 0.5,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 generations: int = 10
                 ):

        self.model = TargetModel(model_name)
        self.population_size = population_size
        self.elite_size = int(elite_rate * self.population_size)
        print(self.elite_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations

        self.population = []
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []

        swanlab.init(
            project="overthink-black",
            name=dataset + str(num),
            config={
                "model_name": model_name,
                "population_size": population_size,
                "elite_rate": elite_rate,
                "crossover_rate": crossover_rate,
                "mutation_rate": mutation_rate,
                "generations": generations,
            }
        )

    def count_tokens(self, chat_usage) -> int:
        if chat_usage is None:
            return 0

        if hasattr(chat_usage, "total_tokens"):
            return chat_usage.total_tokens
        elif hasattr(chat_usage.completion_tokens_details, "reasoning_tokens") and hasattr(chat_usage.completion_tokens_details, "answering_tokens") :
            print('reasoning tokens')
            return chat_usage.completion_tokens
        else:
            return 0

    def evaluate_fitness(self, population) -> List:
        """评估个体的适应度（LLM回复的token数量）"""
        try:
            prompts = [ind.to_prompt() for ind in population]
            results = self.model.agenerate(prompts)
            if results is None:
                return []
            fitness = 0
            token_count = []
            for i, result in enumerate(results):
                _, chat_usage = result
                fitness = self.count_tokens(chat_usage)
                for word in overthink_tokens:
                    fitness = fitness + _.count(word) * 10
                for word in attitude_tokens:
                    fitness = fitness + _.count(word) * 10
                population[i].fitness = fitness
                token_count.append(fitness)
            return token_count
        except Exception as e:
            print(f"评估适应度时出错: {e}")
            return []

    def evaluate_population(self):
        """评估整个种群的适应度"""
        cnt = 0
        sum_fitness = 0
        fitnesses = self.evaluate_fitness(self.population)
        with open(flie_result_path, 'a', encoding='utf-8') as file:
            for individual in self.population:
                print(f'id:{cnt},fitness:{individual.fitness}',file=file)
                sum_fitness = sum_fitness + individual.fitness
                cnt += 1
                if individual.fitness > self.best_fitness:
                    self.best_fitness = individual.fitness
                    self.best_individual = individual.copy()
        with open(flie_all_path, 'a', encoding='utf-8') as file:
            print(f'average:{sum_fitness/cnt}',file=file)

    def roulette_wheel_selection(self, num_parents: int) -> List[Problem]:
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

    def crossover(self, parent1: Problem, parent2: Problem) -> Tuple[Problem, Problem]:
        """交叉操作：交换两个问题的前提或问题"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < self.crossover_rate:
            n = min(len(parent1.son_questions),len(parent2.son_questions))
            num_groups = (n + 1) // 2  # 计算组数
            for group_index in range(1, num_groups + 1):
                start = (group_index - 1) * 2
                end = min(start + 2, n)
                if group_index % 2 == 0:  # 偶数组号进行交换
                    for j in range(start, end):
                        parent1.son_questions[j], parent2.son_questions[j] = parent2.son_questions[j], parent1.son_questions[j]
            child1.son_questions, child2.son_questions = parent2.son_questions.copy(), parent1.son_questions.copy()


        return child1, child2

    def mutate(self, individual: Problem) -> Problem:
        """变异操作：随机消去前提或从其他个体添加前提"""
        mutated = individual.copy()

        if random.random() < self.mutation_rate:
            mutation_type = random.choice(["remove_son_questions", "add_son_questions"])

            if mutation_type == "remove_son_questions" and len(mutated.son_questions) > 0:
                # 消去前提
                mutated.son_questions.pop(random.randint(0, len(mutated.son_questions) - 1))

            elif mutation_type == "add_premise" and len(self.population) > 1:
                # 从其他个体添加前提
                other_individual = random.choice(self.population)
                if other_individual.son_questions:
                    new_son_questions = random.choice(other_individual.son_questions)
                    if  new_son_questions not in mutated.son_questions:
                        mutated.son_questions.append( new_son_questions)

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

    def run(self, initial_problems: List[Problem]) -> Problem:
        """运行遗传算法"""
        # 初始化种群
        self.population = [problem.copy() for problem in initial_problems]

        # 如果初始种群不够大，复制一些个体

        print(f"开始遗传算法优化，种群大小: {self.population_size}, 代数: {self.generations}")

        # 进化循环
        for generation in range(self.generations):
            print(f"\n=== 第 {generation + 1} 代 ===")
            with open(flie_all_path, 'a', encoding='utf-8') as file:
                print(f'epoch:{generation+1}', file=file)
            with open(flie_result_path, 'a', encoding='utf-8') as file:
                print(f'epoch:{generation+1}', file=file)
            self.evolve_generation()

            # 记录当前代的最佳适应度
            current_best_fitness = max(ind.fitness for ind in self.population)
            self.fitness_history.append(current_best_fitness)

            print(f"当前代最佳适应度: {current_best_fitness}")
            print(f"历史最佳适应度: {self.best_fitness}")
            swanlab.log({
                "current_best_fitness": current_best_fitness,
                "best_fitness": self.best_fitness,
            })
            if self.best_individual:
                print(f"最佳个体: {self.best_individual}")

        # 确保返回最佳个体，如果没有则返回种群中最好的
        if self.best_individual is None:
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            return self.population[0]
        return self.best_individual

if __name__ == "__main__":
    # 创建示例问题
    parser = argparse.ArgumentParser()
    output_len = 16384
    parser.add_argument(
        "--function-descript",
        type=str,
        default="You are an expert in mathematics. Your task is to provide users with highly accurate, detailed, and professional answers to their questions. Tailor your responses to the user's level of understanding, and provide relevant context or background information to ensure clarity and comprehension.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='svamp',
    )
    arg = parser.parse_args()
    file_path = './data/'+ arg.dataset +'/' + arg.dataset +'_test.jsonl'
    file_out_path = './data/'+ arg.dataset +'/' + arg.dataset + '_10' +'_change.jsonl'
    flie_result_path = './data/'+ arg.dataset +'/' + arg.dataset +'_result.json'
    flie_all_path = './data/'+ arg.dataset +'/' + arg.dataset +'_all.json'
    file_final_path = './data/' + arg.dataset + '/' + arg.dataset + '_final.jsonl'

    problems = read_questions_from_jsonl(file_out_path)
    n = len(problems)
    print(n)
    for i in range(0,4):
        initial_problems = []
        for j in range(10):
            index = (i + j) % n
            problem_str = problems[index]
            str_problem = problem_str.split(',')
            one_problem = Problem(str_problem)
            initial_problems.append(one_problem)

        print("初始问题:")
        for x, problem in enumerate(initial_problems):
            print(f"{x + 1}. {problem}")

        # 创建遗传算法实例
        ga = GeneticAlgorithm(
            dataset=arg.dataset,
            num = i,
            model_name="deepseek-reasoner",
            population_size=len(initial_problems),
            elite_rate=0.4,
            crossover_rate=0.8,
            mutation_rate=0.1,
            generations=10
        )

        # 运行算法
        best_problem = ga.run(initial_problems)

        print("\n" + "=" * 50)
        print(f"最佳问题: {best_problem}")
        print(f"最佳适应度: {ga.best_fitness}")
        print(f"最佳问题提示: {best_problem.to_prompt()}")