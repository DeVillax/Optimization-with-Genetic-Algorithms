import numpy as np
import random


def read_network(filename):
    """
    Read the network

    filename: (String) Name of the file 

    return: (dict) Network 
    """
    network = {}
    nodes = 0
    links = 0
    matrix = []
    degrees = [] # K Value in our equation
    
    with open(filename) as data:
        nodes = int(data.readline().split()[1])
        matrix = np.zeros((nodes,nodes),dtype=int)

        # Skip non relevant data
        for i in range(nodes):
            data.readline()
        
        # Skip Edges line
        data.readline()

        # Read Links data
        remaining_lines = data.readlines()
        for line in remaining_lines:
            origin, destination, _ = line.split()
            origin = int(origin)
            destination = int(destination)
            # print(origin)
            # print(destination)
            matrix[origin-1][destination-1] = 1
            matrix[destination-1][origin-1] = 1
            links += 1
        
    for line in matrix:
        degrees.append(sum(line))
    
    network["num_nodes"] = nodes
    network["num_links"] = links
    network["degrees"] = degrees
    network["matrix"] = matrix
    return network

def calculate_modularity(network, clusters):
    """
    Calculates the modularity
    """
    modularity = 0
    two_L = sum(network["degrees"]) # 2L in formula 2L = sum(K)
    
    for i in range(network["num_nodes"]):
        for j in range(network["num_nodes"]):
            first = network["matrix"][i][j] - ((network["degrees"][i]*network["degrees"][j])/two_L)
            second = clusters[i] * clusters[j] + ((1-clusters[i])*(1-clusters[j]))
            modularity += first * second
    return modularity/two_L
        
def calculate_fitness(network, chromosome):
    """
    We can simply take the absolute value of the modularity to be our fitness as we basically
    need to convert it to positive values
    """
    return abs(calculate_modularity(network, chromosome))
    
def initial_population(network, num_individuals):
    """
    Initialize population randomly

    network: (dictionary) Loaded network
    num_individuals: (Integer) Number of indivials of the population

    Return: (Matrix) Population
    """
    population = []
    for x in range(num_individuals):
        chromosomes = np.random.randint(2, size=network["num_nodes"])
        population.append(chromosomes)
    return population
       
"""
def rank(fitness):
    sorted_fitness = sorted(fitness) # The indices represent the ranks
    sum_ranks = sum([x for x in range(1,len(fitness)+1)])
    sum_fitness= (sum(sorted_fitness))
    probs = [sorted_fitness[x]/sum_fitness for x in range(len(fitness))]
    print(probs)
"""

def tournament(fitness, population):
    """
    It will return two parents which will be our Cx, and Cy
    """
    selection = []
    aux = fitness.copy()
    K = 5 # Num of individuals to select randomly
    for x in range(2):
        individuals = random.sample(list(enumerate(aux)), K)
        fittest = max(individuals, key=lambda x:x[1])[0]
        selection.append(fittest)
        del aux[fittest]
    
    return selection
        
def crossover_one(parent1, parent2):
    """
    One-point crossover
    """
    ran_position = random.randint(1,len(parent1)-1)

    child1 = []
    child2 = []
    
    half1_parent1 = parent1[:ran_position]
    half2_parent1 = parent1[ran_position:]


    half1_parent2 = parent2[:ran_position]
    half2_parent2 = parent2[ran_position:]

    child1.extend(half1_parent1)
    child1.extend(half2_parent2)
    
    child2.extend(half1_parent2)
    child2.extend(half2_parent1)
    
    return child1, child2

def crossover_two(parent1, parent2):
    """
    Two-point crossover
    """

    child1 = []
    child2 = []

    rand1 = random.randint(1, len(parent1)-1)
    rand2 = random.randint(1, len(parent1)-1)
    
    while(rand1 == rand2):
        rand2 = random.randint(1, len(parent1)-1)
    rand = sorted([rand1, rand2])
    
    p1_swap = parent1[rand[0]:rand[1]]
    p2_swap = parent2[rand[0]:rand[1]]

    child1.extend(parent1[:rand[0]])
    child2.extend(parent2[:rand[0]])

    child1.extend(p2_swap)
    child2.extend(p1_swap)

    if rand[1]-len(parent1) != 0:
        child1.extend(parent1[rand[1]-len(parent1):])
        child2.extend(parent2[rand[1]-len(parent1):])
    

    return child1, child2

def mutation_one(individual):
    """
    Select random gene and flip its value
    """
    index, gene = random.choice(list(enumerate(individual)))
    
    if gene == 0:
        individual[index] = 1
        return individual
    else:
        individual[index] = 0
        return individual

def mutation_two(individual, mutation):
    for index,gene in enumerate(individual):
        if (random.uniform(0, 1) < mutation):
            if gene == 0:
                individual[index] = 1  
            else:
                individual[index] = 0
    return individual
    
    


if __name__ == "__main__":

    # Initial parameters
    population_size = 50
    num_generations = 200

    # Load Network
    filename = "256_4_4_4_13_18_p"
    network = read_network(f".\\A4-networks\\{filename}.net")

    # Initialize population P
    population = initial_population(network, population_size)

    # Evaluate fitness of all individuals in P
    fitness = []
    for individual in population:
        f = calculate_fitness(network, individual)
        fitness.append(f)


    best = 0
    # For loops
    for generation in range(num_generations):
        print(f"Generation {generation}")
        aux_population = []
        for pair in range(int(population_size/2)):
            # Select two individuals Cx, Cy from P
            parent1, parent2 = tournament(fitness, population)

            # Crossover
            child1, child2 = crossover_one(population[parent1].tolist(), population[parent2].tolist())

            # Mutation
            mutated1 = mutation_one(child1)
            mutated2 = mutation_one(child2)
            
            aux_population.append(np.array(mutated1))
            aux_population.append(np.array(mutated2))
        
        fittest = max(list(enumerate(fitness)), key=lambda x:x[1])
        if fittest[1] > best:
            best = fittest[1]
            loc = fittest[0]
            print(f"Fittest:{fittest}")
            sol = population[loc]

        # Elitism: add best fitted individuals of P to P' 
        # For now we will just copy the fittest invidual
        # fittest = max(list(enumerate(fitness)), key=lambda x:x[1])[0]
        # lowest = min(list(enumerate(fitness)), key=lambda x:x[1])[0]

        # aux_fitness =[]
        # for individual in aux_population:
        #     f = calculate_fitness(network, individual)
        #     aux_fitness.append(f)

        # lowest = min(list(enumerate(aux_fitness)), key=lambda x:x[1])[0]
        # aux_population[lowest] = population[fittest]

        fittest = sorted(fitness)[-5:]
        f = {fitness[x]:x for x in range(len(fitness))}
        indices = [f[x] for x in fittest]
        aux_fitness =[]
        for individual in aux_population:
            f = calculate_fitness(network, individual)
            aux_fitness.append(f)

        lowest = sorted(aux_fitness)[:5]
        f2 = {aux_fitness[x]:x for x in range(len(aux_fitness))}
        indices2 = [f2[x] for x in lowest]

        #print(aux_population)
        for x in range(5):
            aux_population[indices2[x]] = population[indices[x]]
        #print(aux_population)
        
        aux_fitness =[]
        for individual in aux_population:
            f = calculate_fitness(network, individual)
            aux_fitness.append(f)
        
        population = aux_population
        
        # Evaluate fitness of all individuals in P
        fitness = []
        for individual in population:
            f = calculate_fitness(network, individual)
            fitness.append(f)
    
    print(f"Best modularity found: {best}")
    #print(sol)
    #print(calculate_modularity(network, sol))

    # Save results
    with open(f".\\results\\{filename}.clu","w") as fr:
        fr.write(f"*Vertices {network['num_nodes']}\n")
        for gene in sol:
            fr.write(f"{gene}\n")