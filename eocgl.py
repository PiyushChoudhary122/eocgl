import random
import numpy as np
from deap import base, creator, tools, algorithms

# Create the optimization problem's classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# Define the simple linear regression problem
def linear_regression(individual, x):
    a, b = individual[0], individual[1]
    y_predicted = a * x + b
    return sum((y_predicted - x) ** 2),  # Sum of squared errors as the fitness


# Create a toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)  # Coefficients 'a' and 'b'
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", linear_regression, x=np.array([1, 2, 3, 4, 5]), )  # Sample input data
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation

# Genetic Algorithm parameters
population_size = 100
num_generations = 50
crossover_prob = 0.7
mutation_prob = 0.2

# Create an initial population
population = toolbox.population(n=population_size)

# Evaluate the fitness of the initial population
fitness_values = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitness_values):
    ind.fitness.values = fit

# Genetic Algorithm
for gen in range(num_generations):
    offspring = algorithms.varOr(population, toolbox, lambda_=population_size, cxpb=crossover_prob, mutpb=mutation_prob)

    # Evaluate the fitness of the offspring
    fitness_values = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fitness_values):
        ind.fitness.values = fit

    # Select the next generation using tournament selection
    population = tools.selBest(population + offspring, k=population_size)

# Find the best individual (linear regression coefficients)
best_individual = tools.selBest(population, k=1)[0]
best_a, best_b = best_individual[0], best_individual[1]
print("Best Linear Regression Model:")
print(f"y = {best_a} * x + {best_b}")

# Make predictions using the best model
input_data = np.array([6, 7, 8, 9, 10])
predictions = best_a * input_data + best_b
print("Predictions for input data [6, 7, 8, 9, 10]:", predictions)
