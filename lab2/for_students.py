from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    #my_dict = {}
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        #my_dict[tuple(individual)] = individual_fitness
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness/total_fitness for fitness in fitness_values]
    chosen_indices = random.choices(range(len(population)), probabilities, k=len(population))
    chosen_individuals = [population[index] for index in chosen_indices]
    #for individual in chosen_individuals:
       #print(individual)
    return chosen_individuals

def mutation(individual,mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = not individual[i]
    return individual


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1
mutation_rate=0.05

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)

for o in range(generations):
    population_history.append(population)
    # TODO: implement genetic algorithm

    values=[]
    for organism in population:
        f = fitness(items, knapsack_max_capacity, organism)
        values.append(f)
    chosen = roulette_wheel_selection(population, values)       #Selekcja

    children = []
    for i, key in enumerate(chosen):     #epiccrossover
        if i < 49:
            j = random.randint(0, len(chosen) - 1)
            k = random.randint(0, len(chosen) - 1)
            firstHalf = chosen[j]
            secondHalf = chosen[k]
            firstChild = firstHalf[:13] + secondHalf[-13:]
            secondChild = secondHalf[:13] + firstHalf[-13:]
            children.append(firstChild.copy())
            children.append(secondChild.copy())

    # for child in children:        #mutowanie dzieci
    #     for i in range(3):
    #         i = random.randint(0, len(child) - 1)
    #         child[i] = not child[i]

    for i in range(len(children)):      #mutowanie dzieci
        children[i] = mutation(children[i], mutation_rate)

    for _ in range(0, n_elite):     #eliarnie wybrani osobnicy SIC
        max_index = values.index(max(values))
        children.append(population[max_index])
        del population[max_index-1]
        del values[max_index - 1]
    population = children
    print(len(population))

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.',color='blue')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
