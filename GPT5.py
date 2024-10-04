import numpy as np
import matplotlib.pyplot as plt

# Isti k..

# Define environment
class Environment:
    def __init__(self):
        # Define state space
        self.state_space = [('A', 1), ('A', 2), ('A', 3), ('A', 4), ('A', 5),
                            ('B', 1), ('B', 3), ('B', 5)]

        # Start state
        self.start_state = ('A', 1)
        self.current_state = self.start_state

        # Terminal states and rewards
        self.terminal_states = {('B', 1): -1, ('B', 3): -1, ('B', 5): 3}

        # Actions
        self.actions = ['up', 'down', 'left', 'right']

        # Rows and columns
        self.rows = ['A', 'B']
        self.columns = [1, 2, 3, 4, 5]

        # Initialize Q-values
        self.Q = {state: {action: 0 for action in self.actions} for state in self.state_space}

    # Step function to simulate one step in the environment
    def step(self, action):
        state = self.current_state

        # Check if state is terminal
        if state in self.terminal_states:
            return state, self.terminal_states[state], True

        # Determine next state based on stochastic rules
        next_state = self.stochastic_move(state, action)

        if next_state in self.terminal_states:
            self.current_state = next_state
            return next_state, self.terminal_states[next_state], True

        # Update current state
        self.current_state = next_state

        # Return next state, reward, and done flag
        return next_state, 0, False

    # Function to add stochasticity to agent's moves
    def stochastic_move(self, state, action):
        main_prob = 0.6
        side_prob = (1 - main_prob) / 2

        row, col = state

        # Define possible move directions
        if action == 'up':
            moves = ['up', 'left', 'right']
        elif action == 'down':
            moves = ['down', 'left', 'right']
        elif action == 'left':
            moves = ['left', 'up', 'down']
        elif action == 'right':
            moves = ['right', 'up', 'down']
        else:
            raise ValueError("Invalid action")

        # Choose actual move based on probabilities
        chosen_move = np.random.choice(moves, p=[main_prob, side_prob, side_prob])

        # Determine new state based on chosen move
        if chosen_move == 'up':
            next_state = (self.rows[self.rows.index(row) - 1] if self.rows.index(row) > 0 else row, col)
        elif chosen_move == 'down':
            next_state = (self.rows[self.rows.index(row) + 1] if self.rows.index(row) < len(self.rows) - 1 else row, col)
        elif chosen_move == 'left':
            next_state = (row, col - 1 if col > 1 else col)
        elif chosen_move == 'right':
            next_state = (row, col + 1 if col < 5 else col)
        else:
            next_state = state

        # Check if next state is valid
        if next_state not in self.state_space:
            next_state = state

        return next_state

    # Function to reset environment to initial state
    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    # Q-learning algorithm
    def q_learning(self, num_episodes, alpha, gamma, epsilon, test_interval):
        rewards_during_learning = []
        q_values_during_learning = []

        for e in range(num_episodes):
            state = self.reset()
            done = False
            episode_reward = 0

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.actions)
                else:
                    action = max(self.Q[state], key=self.Q[state].get)

                next_state, reward, done = self.step(action)
                best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
                self.Q[state][action] += alpha * (reward + gamma * self.Q[next_state][best_next_action] - self.Q[state][action])
                state = next_state

                episode_reward += reward

            # Adjust epsilon and alpha
            epsilon = np.log(e + 2) / (e + 2)
            alpha = np.log(e + 2) / (e + 2)

            # Periodic testing
            if (e + 1) % test_interval == 0:
                average_reward = self.test_policy(10)
                rewards_during_learning.append((e + 1, average_reward))
                q_values_during_learning.append(self.get_q_values_snapshot())

            print(f"Episode: {e + 1}, Total reward: {episode_reward}")

        return rewards_during_learning, q_values_during_learning

    # Plot V-values
    def plot_v_values(self, Q, title):
        V = {state: max(actions.values()) for state, actions in Q.items()}
        V_matrix = np.zeros((2, 5))
        for state, value in V.items():
            row = 0 if state[0] == 'A' else 1
            col = state[1] - 1
            V_matrix[row, col] = value

        fig, ax = plt.subplots()
        cax = ax.matshow(V_matrix, cmap='coolwarm')
        for (i, j), val in np.ndenumerate(V_matrix):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
        plt.title(title)
        plt.show()

    # Function to get a snapshot of current Q-values
    def get_q_values_snapshot(self):
        snapshot = {}
        for state in self.state_space:
            if state not in self.terminal_states:
                snapshot[state] = max(self.Q[state].values())
        return snapshot

    # Plot average rewards during learning
    def plot_rewards_during_learning(self, rewards_during_learning, title):
        episodes, rewards = zip(*rewards_during_learning)
        plt.plot(episodes, rewards, label='Average Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title(title)
        plt.legend()
        plt.show()

    # Plot Q-values during learning
    def plot_q_values_during_learning(self, q_values_during_learning, title):
        for i, q_values in enumerate(q_values_during_learning):
            V_matrix = np.zeros((2, 5))
            for state, value in q_values.items():
                row = 0 if state[0] == 'A' else 1
                col = state[1] - 1
                V_matrix[row, col] = value
            fig, ax = plt.subplots()
            cax = ax.matshow(V_matrix, cmap='coolwarm')
            for (i, j), val in np.ndenumerate(V_matrix):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
            plt.title(f'{title} - Snapshot {i + 1}')
            plt.show()

    # Function to test current policy over a number of episodes
    def test_policy(self, num_episodes):
        total_rewards = 0
        for _ in range(num_episodes):
            state = self.reset()
            done = False
            episode_reward = 0
            while not done:
                action = max(self.Q[state], key=self.Q[state].get)
                next_state, reward, done = self.step(action)
                print(f"Action: {action}, New state: {next_state}, Reward: {reward}, Done: {done}")
                episode_reward += reward
                state = next_state
            total_rewards += episode_reward
        return total_rewards / num_episodes

# Main script
env = Environment()
num_episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1
test_interval = 50

rewards_during_learning, q_values_during_learning = env.q_learning(num_episodes, alpha, gamma, epsilon, test_interval)
env.plot_rewards_during_learning(rewards_during_learning, 'Average Reward during Learning with Variable Learning Rate')
env.plot_q_values_during_learning(q_values_during_learning, 'Q-values during Learning')

# Test with different gamma
gamma = 0.999
env.Q = {state: {action: 0 for action in env.actions} for state in env.state_space}  # Reset Q-values
rewards_during_learning, q_values_during_learning = env.q_learning(num_episodes, alpha, gamma, epsilon, test_interval)
env.plot_rewards_during_learning(rewards_during_learning, 'Average Reward during Learning with High Gamma (0.999)')
env.plot_q_values_during_learning(q_values_during_learning, 'Q-values during Learning with High Gamma (0.999)')
