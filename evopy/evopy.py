import numpy as np

from evopy.individual import Individual


def run(fitness_function, individual_length, warm_start=None, generations=100, population_size=30, num_children=1, mean=0, std=1, maximize=False):
    if individual_length == 0:
        return

    if not warm_start:
        warm_start = np.zeros(individual_length)

    population = _init_population(population_size, individual_length, mean, std, warm_start)
    best = sorted(population, reverse=maximize, key=lambda individual: fitness_function(individual))[0]
    for i in range(generations):
        children = [parent.reproduce() for _ in range(num_children) for parent in population]
        population = sorted(children + population, reverse=maximize, key=lambda individual: fitness_function(individual))
        population = population[:population_size]

        if not maximize:
            best = population[0] if population[0].fitness > best.fitness else best
        else:
            best = population[0] if population[0].fitness < best.fitness else best
    return best


def _init_population(population_size, individual_length, mean, std, offset):
    return [Individual(offset + np.random.normal(loc=mean, scale=std, size=(len(individual_length))), np.random.randn()) for _ in range(population_size)]
