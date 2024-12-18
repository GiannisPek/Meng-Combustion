import numpy as np
import matplotlib.pyplot as plt
import math
import pygad
from Engine_sim import EngineMechanism


def fitness_func(ga_instance, solution, solution_idx):
    rpm = solution[0]
    theta = solution[1]
    engine = EngineMechanism(rpm, theta)
    Pmax, Tmax, penalty, theta = engine.run_full_simulation()

    if penalty:
        pp = 100*(theta)**2
    else:
        pp =0 
    
    obj = Pmax/100 + Tmax/300 - pp
    return obj

fitness_function = fitness_func

def on_start(ga_instance):
    print("on_start()")

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

sol_per_pop = 35
num_generations = 25
num_genes = 2
gene_space = [
    {'low': 60, 'high': 100},  # Gene 1: Boundary [0, 10]
    {'low': -5, 'high': 2},  # Gene 2: Boundary [-5, 5]
]


ga_instance = pygad.GA(num_generations = num_generations,
                    gene_space= gene_space,
                    num_parents_mating=int(np.ceil(0.2*sol_per_pop)),
                    fitness_func=fitness_function,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    mutation_type = 'random',
                    save_solutions = True
                    )

ga_instance.run()

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best Solution: ", solution)
print("Fitness of Best Solution: ", solution_fitness)






