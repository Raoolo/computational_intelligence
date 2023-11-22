### Collaborated with Donato Lanzilloti on the solution


### Overview

This code is designed to solve the Nim game problem by utilizing Evolutionary Strategies (ES). The Nim game is a mathematical game of strategy in which two players take turns removing objects from distinct heaps. In this implementation, we explore various strategies with the objective of identifying the optimal one for victory.

### Strategies

The code includes multiple strategies with initially equal probabilities of being chosen:
- Pure Random: Randomly selects moves.
- Gabriele: Chooses the maximum number of objects from the lowest row.
- Raul: Removes one object from the row with the maximum number of objects.
- Optimal: Based on the nim-sum calculation, which is crucial for determining winning moves.

### Starting Solution

The initial probability distribution for the use of each strategy is set to 25%.
Key Components
- Nim Class: Represents the state of the game with methods to make moves.
- Strategy Functions: Define how a move is chosen based on the current state of the game.
- Evolutionary Approach: Assigns and adjusts probabilities to each strategy based on their success, seeking to maximize the probability of the optimal strategy over time.

### Usage
- Initialize the game with a specified number of rows.
- The eval function plays the game multiple times using different strategies and evaluates their success rates.
- The ES loop generates offspring solutions and selects the best ones based on the game outcomes.
- The process iteratively adjusts strategy probabilities and the standard deviation (σ) of the offspring generation process.
- The final output is the best strategy probability distribution found.
