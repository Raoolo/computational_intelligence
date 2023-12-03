This program implements a genetic algorithm to solve the problem. The main methods are:

### Evaluate population
The program evaluates its genomes only if it is necessary, and to check this it uses a dictionary where it stores the already evaluated genomes as to reduce the number of calls to the fitness function. It then checks for every genomes if it was already evaluated.

### Tournament selection
The program selects a subset of genomes and then picks the best one among them, for a fixed amount of times to build a population. This will be passed to the crossover and mutation function.

### Elitism
The program also selects the best genomes among all of the population to bring on.

### Crossover 
The previously found genomes are passed to a function that crosses the genomes at a random point and creates 2 offsprings from the parents.

### Mutation 
The function, with a certain probability, switches some of the bits of a genome to introduce variability. This is done with NumPy to speed up the operation.

### Running the code
The code is run thanks to a function that creates the instance of the problem, creates a first population and then iterates until the stopping condition is met. In the loop, the first method called is elitism, followed by tournament selection and the output of the latter is then passed to the crossover and finally to the mutation function. The population is then updated to keep track of the offsprings generated.
There are two ways of stopping the code, the first one is based on a fixed number of generations after which the program will stop and return the best found genomes and fitness.
The second version is based on a dynamic termination condition, and it check that the fitness is actually improving through the generations and is not stuck. Once the improvement is neglible the program stops and returns the results.

--- 

## To improve
The results currently seem not great. The reason is still being investigated, and might be due to poor hyperparameter setting. 
