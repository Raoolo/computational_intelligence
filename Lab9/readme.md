This program implements a genetic algorithm to solve the problem. The main implementations are:

### Tournament selection
The program selects a subset of genomes and then picks the best one among them, for a fixed amount of times to build a population. This will be passed to the crossover and mutation function.

### Elitism
The program also selects the best genomes among all of the population to bring on.

### Crossover 
The previously found genomes are passed to a function that crosses the genomes at a random point and creates 2 offsprings from the parents.

### Mutation 
The function, with a certain probability, switches some of the bits of a genome to introduce variability. This is done with NumPy to speed up the operation.

### Termination condition
There are two ways of running the code, the first one is based on a fixed number of generations after which the program will stop and return the best found genomes and fitness.
The second version is based on a dynamic termination condition, and it check that the fitness is actually improving through the generations and is not stuck. Once the improvement is neglible the program stops and returns the results.
