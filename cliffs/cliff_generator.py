import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# PARAMETERS
NUM_POINTS = 80
POOL_COUNT_TARGET = 3
POOL_SIZE_TARGET = 7
FLOOD_PERCENTAGE_TARGET = 0.3

POP_SIZE = 80
GENS = 12000
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

# CREATE FITNESS FUNCTION (Maximize fitness)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# REGISTRATION OF INDIVIDUALS
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.0, 1.0)  # Heights between 0 and 1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_POINTS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# FITNESS FUNCTION
def evaluate(individual):
    heights = np.array(individual)

    flooded_mask = heights < 0.5
    flooded_count = np.sum(flooded_mask)
    flood_percentage = flooded_count / NUM_POINTS

    pool_sizes = []
    current_pool_size = 0
    for h in flooded_mask:
        if h:
            current_pool_size += 1
        else:
            if current_pool_size > 0:
                pool_sizes.append(current_pool_size)
            current_pool_size = 0
    if current_pool_size > 0:
        pool_sizes.append(current_pool_size)

    num_pools = len(pool_sizes)

    smoothness_penalty = np.sum(np.abs(np.diff(heights)))

    # FITNESS SCORE (Higher is better)
    pool_count_score = 4 * -abs(num_pools - POOL_COUNT_TARGET)
    pool_size_score = 0.7 * -sum(abs(size - POOL_SIZE_TARGET) for size in pool_sizes) if pool_sizes else -POOL_SIZE_TARGET
    flood_score = 10 * -abs(flood_percentage - FLOOD_PERCENTAGE_TARGET)

    fitness = pool_count_score + pool_size_score + flood_score - 0.1 * smoothness_penalty

    return (fitness,)


# REGISTER OPERATORS
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.4)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=MUTATION_RATE)
toolbox.register("select", tools.selTournament, tournsize=3)


# MAIN EVOLUTION FUNCTION
def evolve():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)  # Store best solution

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE, ngen=GENS,
                                 stats=stats, halloffame=hof, verbose=True)

    return hof[0]  # Return the best terrain


# PLOT FUNCTION
def plot_terrain(terrain):
    x = np.arange(NUM_POINTS)
    y = np.array(terrain)

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="Terrain Height", color="black")
    plt.fill_between(x, y, 0.5, where=(y < 0.5), color="blue", alpha=0.3, label="Water")

    plt.axhline(y=0.5, color="red", linestyle="--", label="Water Level")
    plt.xlabel("Point Index")
    plt.ylabel("Height")
    plt.legend()
    plt.title("Generated Terrain")
    plt.show()

if __name__ == "__main__":
    best_terrain = evolve()
    plot_terrain(best_terrain)
