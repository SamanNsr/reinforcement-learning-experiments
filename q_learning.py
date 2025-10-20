import numpy as np
import random

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.2):
        self.n_states = n_states
        self.n_actions = n_actions


        self.learning_rate = learning_rate

        # this is discount factor for future rewards in the bellman equation
        self.discount_factor = discount_factor

        # this is the epsilon for the epsilon-greedy policy (we can use upper confidence bound but its not common in Q-learning)
        self.epsilon = epsilon

        # this is our Q-table that we should learn
        self.q = np.zeros((n_states, n_actions), dtype=float)

    def get_action(self, state_index):
        if random.random() < self.epsilon:
            # exploration
            return random.randint(0, self.n_actions - 1)  
        else:
            # exploitation
            return np.argmax(self.q[state_index])
        
    def update(self, state_index, action, reward, next_state_index, is_final_state):
        # We should use approximate methods such as Temporal Difference instead of Dynamic Prgamming and Monte Carlo methods
        # because solving markov decision process exactly is not feasible for large state spaces
        # and dynamic programming requires a model of the environment which is not always available
        # and monte carlo method although is model-free but it requires waiting until the end of the episode to update the values

        # the Q-learning update rule is as follows which follows bellman equation
        # Q[s,a] <- Q[s,a] + learning_rate * (r + discount_factor * max_a' Q[s',a'] * (1-is_final_state) - Q[s,a])

        max_next = 0 if is_final_state else np.max(self.q[next_state_index])
        td_future = reward + self.discount_factor * max_next
        self.q[state_index, action] += self.learning_rate * (td_future - self.q[state_index, action])  

class GridWorld():
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.state = self.start
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 2)]
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action_index):
        x, y = self.state

        action = self.actions[action_index] 
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)

        next_state = (x, y)

        reward = 0
        if next_state in self.obstacles:
            next_state = self.state
            reward = -1
        elif next_state == self.goal:
            reward = 10
        else:
            reward = -0.1

        is_final_state = (next_state == self.goal)
        self.state = next_state
        return next_state, reward, is_final_state
    

if __name__ == "__main__":
    n_episodes = 100
    env = GridWorld(size=5)
    agent = QLearning(n_states=25, n_actions=4)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        is_final_state = False

        while not is_final_state and steps < 200: 
            state_index = state[0] * env.size + state[1]
            action = agent.get_action(state_index)
            next_state, reward, is_final_state = env.step(action)
            next_state_index = next_state[0] * env.size + next_state[1]
            agent.update(state_index, action, reward, next_state_index, is_final_state) 
            
            state = next_state
            total_reward += reward
            steps += 1

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Total Reward: {total_reward}")

        