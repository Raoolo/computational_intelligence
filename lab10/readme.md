# Reinforcement learning for Tic-Tac-Toe game

### Objective
We want to create an agent that can learn the optimal strategy to beat the adversary through iterations of the game.

The games will be based on a 3x3 grid, that can easily be modified, and the agent is the player that has to learn the best strategy by observing the state of the game (the board). To learn we will reward good actions and penalyze wrong ones.

### Q-Values and Q-Table
To represent the expected future reward for an action taken in a given state we will utilize the Q-value and a Q-table. The Q-table instead is a dictionary that contains as key the current state and as value another dictionary, saving the Q-value for each possible action from the current state.

### Choose actions
The agent explores the possible actions in two ways: 
- one is the random approach based on the epsilon-greedy strategy, meaning based on the decaying value of epsilon it will explore random solutions
- the other is the exploitation, meaning that it will choose the best-known actions for the possible move


### Training and testing
The agent will play a number of games (in the example, 10000) to learn the best approach to play the game against a random agent. Then, it is tested 1000 times and we can see that it never loses and at most draws against them very few times. Each game, the starting player is chosen randomly to avoid overfitting of certain strategies.
There is also the `print_board` method that is used to actually see the board and how the players are playing, but in this case it is commented to avoid the output in the terminal.

### Improvements
The agents could learn to play against a better player (not a random one).
