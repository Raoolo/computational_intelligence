# Reinforcement Learning and MinMax for Quixo

### Overall view
In this project a Reinforcement Learning agent is capable of playing the Quixo game against a random player (or even another RL agent if wanted). The core of the project is the RL algorithm that allows the agent to make strategic moves based on the current state of the game. The agent learns through episodes of games, improving its strategy by updating the Q-table that maps the state-action to the rewards.

### Features
- Implementation of Q-learning algorithm.
- Utilization of MinMax algorithm as secondary strategy.
- Exploration/Exploitation balanced with epsilon-greedy approach.
- Training and evaluation against a random player.

### Training the Q-table
To train the Q-table, the board has been modified in different ways to increase the diversity of situations the Q-learning algorithm learns, generating up to eight different variations for each state by rotating and mirroring the board. 
The rewards are based on the feedback of the action, giving more importance to actions that lead to the win of the game.

### Role of MinMax
For game states where the best action is not obvious (i.e. winning moves), MinMax provides a way to evaluate the possible moves by considering a fixed number of potential future states, simulating moves for both the agent and the opponent, and trying to maximize the player's advantage. This allows the Q-learning agent to consider also the paths that lead to the win, even when the win is not immediate.
