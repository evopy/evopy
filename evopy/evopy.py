"""Module used for the execution of the evolutionary algorithm."""
import numpy as np

from evopy.individual import Individual


#pylint: disable=R0913
def run(fitness_function, individual_length, warm_start=None, generations=100, population_size=30,
        num_children=1, mean=0, std=1, maximize=False):
    """Run the evolutionary strategy algorithm.

    :param fitness_function: the fitness function on which the individuals are evaluated
    :param individual_length: the length of each individual
    :param warm_start: the individual to start from
    :param generations: the number of generations to execute
    :param population_size: the population size of each generation
    :param num_children: the number of children generated per parent individual
    :param mean: the mean
    :param std: the standard deviation
    :param maximize: whether the fitness function should be minimized or maximized
    :return: the best individual found during execution of the algorithm
    """
    if individual_length == 0:
        return None

    if warm_start is None:
        warm_start = np.zeros(individual_length)

    population = _init_population(population_size, individual_length, mean, std, warm_start)
    best = sorted(population, reverse=maximize,
                  key=lambda individual: individual.evaluate(fitness_function))[0]
    for _ in range(generations):
        children = [parent.reproduce() for _ in range(num_children) for parent in population]
        population = sorted(children + population, reverse=maximize,
                            key=lambda individual: individual.evaluate(fitness_function))
        population = population[:population_size]

        if not maximize:
            best = population[0] if population[0].fitness > best.fitness else best
        else:
            best = population[0] if population[0].fitness < best.fitness else best
    return best.weights


def _init_population(population_size, individual_length, mean, std, offset):
    return [
        Individual(offset + np.random.normal(loc=mean, scale=std, size=individual_length),
                   np.random.randn()) for _ in range(population_size)]
