import math
from random import randint, choices, sample
from typing import Callable
import matplotlib.pyplot as plt

import numpy as np
from deap import base, creator, tools, algorithms

type Agent = Callable[[list[int], list[int]], int]


def always_coop(own: list[int], opp: list[int]) -> int:
    return 0

def always_betray(own: list[int], opp: list[int]) -> int:
    return 1

def titfortat(own: list[int], opp: list[int]) -> int:
    if len(opp) == 0:
        return 0
    return opp[-1]

def jesuswheel(own: list[int], opp: list[int]) -> int:
    return randint(0,1)

def cooperate_with_probability(own: list[int], opp: list[int]) -> int:
        return 0 if randint(0, 100) < 0.4 * 100 else 1  # Cooperate with probability `p`

def tit_for_two_tats(own: list[int], opp: list[int]) -> int:
    if len(opp) == 0:
        return 0  # Cooperate first turn
    if len(opp) >= 2 and opp[-1] == 1 and opp[-2] == 1:
        return 1  # Betray if opponent betrayed twice in a row
    return 0  # Otherwise, cooperate

def win_stay_lose_shift(own: list[int], opp: list[int]) -> int:
    if len(own) == 0:
        return 0  # Cooperate first turn
    if own[-1] == 0 and opp[-1] == 0:
        return 0  # Cooperate if last result was mutual cooperation
    if own[-1] == 1 and opp[-1] == 1:
        return 1  # Betray if last result was mutual betrayal
    return 0  # Otherwise, cooperate

def generate_seed(size: int) -> list[int]:
    return list(randint(0,1) for _ in range(size))


def evaluate_result(result: (int, int)) -> float:
    if result[0] == 0:
        return 0
    if result[1] == 0:
        return 2147483647
    return float(result[0]) / result[1]

def seed_to_trainee(seed: list[int]):
    def trainee(own: list[int], opp: list[int]) -> int:
        if len(own) == 0:
            return 0

        opp_aggresiveness = sum(opp)
        if opp_aggresiveness >= (len(seed) - 4):  # why you so aggresive bruh, you capped my seed
            return 1

        result = seed[opp_aggresiveness]
        threshold = 0.25 + (
                    int("".join(map(str, seed[-4:])), 2) / 15) * 0.75  # map the last 4 bits to a 0.25 - 1.0 range
        winrate = 0

        for turn in range(len(own)):
            if own[turn] == opp[turn]:
                if own[turn] == 0:
                    winrate += 1
            else:
                if own[turn] > opp[turn]:
                    winrate += 2

        winratio = winrate / len(own)

        if winratio >= threshold:
            return result
        else:
            return 1 - result
    return trainee

def kiku_the_nights_flwr_1(own: list[int], opp: list[int]):
    temp = seed_to_trainee([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1])
    return temp(own, opp)

#0.472
def kiku_the_nights_flwr_2(own: list[int], opp: list[int]):
    temp = seed_to_trainee([1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    return temp(own, opp)

#0.3664
def kiku_the_nights_flwr_3(own: list[int], opp: list[int]):
    temp = seed_to_trainee([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0])
    return temp(own, opp)

#0.35366
def kiku_the_nights_flwr(own: list[int], opp: list[int]):
    seed = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    if len(own) == 0:
        return 0

    opp_aggresiveness = sum(opp)
    if opp_aggresiveness >= (len(seed) - 4):  # why you so aggresive bruh, you capped my seed
        return 1

    result = seed[opp_aggresiveness]
    threshold = 0.25 + (
            int("".join(map(str, seed[-4:])), 2) / 15) * 0.75
    winrate = 0

    for turn in range(len(own)):
        if own[turn] == opp[turn]:
            if own[turn] == 0:
                winrate += 1
        else:
            if own[turn] > opp[turn]:
                winrate += 2

    winratio = winrate / len(own)

    if winratio >= threshold:
        return result
    else:
        return 1 - result


def evaluate_agent(agent_seed: list[int]):
    total_ratio = 0
    all_opponents = [always_coop, always_betray, titfortat, kiku_the_nights_flwr_1, kiku_the_nights_flwr_2, kiku_the_nights_flwr_3 ,cooperate_with_probability, tit_for_two_tats, win_stay_lose_shift]
    num_all_opponents = len(all_opponents)
    trainee = seed_to_trainee(agent_seed)
    num_opponents = math.ceil(num_all_opponents - 3)
    opponents = sample(all_opponents, k=num_opponents)

    for opponent in opponents:
        result = play_game(
            trainee,  # Our agent (sum history index lookup)
            opponent,
            randint(60,170)
        )
        total_ratio += evaluate_result(result)  # Accumulate ratio
    return (total_ratio / num_opponents,)  # Return average ratio

# 0.639083
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]

def play_game(agent1: Agent, agent2: Agent, rounds: int) -> (int, int):
    history1 = []
    history2 = []

    for turn in range(rounds):
        first = agent1(history1, history2)
        second = agent2(history2, history1)
        history1.append(first)
        history2.append(second)

    score_1 = 0
    score_2 = 0

    for turn in range(rounds):
        if history1[turn] == history2[turn]:
            if history1[turn] == 0:
                score_1 += 1
                score_2 += 1
            else:
                score_1 += 2
                score_2 += 2
        else:
            if history1[turn] < history2[turn]:
                score_1 += 3
            else:
                score_2 += 3
    return score_1, score_2


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool", randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=204)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_agent)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=5)


if __name__ == '__main__':
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(5)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=15,
        lambda_=100,
        cxpb=0.35,
        mutpb=0.25,
        ngen=6000,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    generations = logbook.select("gen")
    min_fitnesses = logbook.select("min")

    # Plot the minimum fitness over time
    plt.figure(figsize=(10, 5))
    plt.plot(generations, min_fitnesses, marker="o", linestyle="-", color="blue", label="Min Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Lower is Better)")
    plt.title("Evolution of Minimum Fitness Over Generations")
    plt.legend()
    plt.grid()
    plt.show()

    # Print the lowest fitness recorded
    best_fitness = hof[0].fitness.values[0]
    print(f"Lowest fitness found: {best_fitness}")

    print(hof[0])

