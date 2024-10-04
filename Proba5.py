import numpy as np
import matplotlib.pyplot as plt

# Izbacena polja (B,2) i (B,4)
# Kao da mi se ucinilo da drugi grafik za prosecnu nagradu nije bio linearan
# Nekad je linearan nekad nije
# Mislim da fali grafikon za prikaz q vrednosti


# Definicija okruženja
class Environment:
    def __init__(self):
        # Definicija mreže stanja
        self.state_space = [('A', 1), ('A', 2), ('A', 3), ('A', 4), ('A', 5),
                            ('B', 1), ('B', 3), ('B', 5)]

        # Početno stanje
        self.start_state = ('A', 1)
        self.current_state = self.start_state # inicijalizujemo trenutno stanje agenta

        # Terminalna stanja i nagrade
        self.terminal_states = {('B', 1): -1, ('B', 3): -1, ('B', 5): 3}

        # Moguce akcije
        self.actions = ['up', 'down', 'left', 'right']

        # Redovi i kolone
        self.rows = ['A', 'B']
        self.columns = [1, 3, 5]  # Isključeni su 2 i 4

        # Inicijalizacija Q-vrednosti
        self.Q = {state: {action: 0 for action in self.actions} for state in self.state_space}

    # Funkcija koja simulira jedan korak u okruženju
    def step(self, action):
        state = self.current_state

        # Provera da li je stanje terminalno
        if state in self.terminal_states:
            return state, self.terminal_states[state], True

        # Određivanje narednog stanja u skladu sa stohastičkim pravilima
        next_state = self.stochastic_move(state, action)

        if next_state in self.terminal_states:
            self.current_state = next_state
            return next_state, self.terminal_states[next_state], True

        # Ažuriranje trenutnog stanja
        self.current_state = next_state

        # Povratak sledećeg stanja, nagrade i informacije da epizoda nije završena
        return next_state, 0, False

    # Funkcija koja uvodi stohastičnost u pokrete agenta
    def stochastic_move(self, state, action):
        main_prob = 0.6
        side_prob = (1 - main_prob) / 2

        row, col = state

        # Definisanje mogućih pravaca kretanja
        if action == 'up':
            moves = ['up', 'left', 'right']
        elif action == 'down':
            moves = ['down', 'left', 'right']
        elif action == 'left':
            moves = ['left', 'up', 'down']
        elif action == 'right':
            moves = ['right', 'up', 'down']
        else:
            raise ValueError("Neispravna akcija")

        # Izbor stvarnog pravca kretanja na osnovu verovatnoća
        chosen_move = np.random.choice(moves, p=[main_prob, side_prob, side_prob])

        # Određivanje novog stanja na osnovu izabranog pravca kretanja
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

        # Provera da li je sledeće stanje validno
        if next_state not in self.state_space:
            next_state = state

        return next_state

    # Funkcija za resetovanje okruženja na početno stanje
    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    # Q-learning algoritam
    # Funkcija za Q-učenje
    def q_learning(self, num_episodes, alpha, gamma, epsilon, test_interval):
        rewards_during_learning = []
        q_values_during_learning = []

        for e in range(num_episodes):
            state = self.reset()
            done = False
            episode_reward = 0  # Dodato: Ukupna nagrada u epizodi

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.actions)
                else:
                    action = max(self.Q[state], key=self.Q[state].get)

                next_state, reward, done = self.step(action)
                best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
                self.Q[state][action] += alpha * (reward + gamma * self.Q[next_state][best_next_action] - self.Q[state][action])
                state = next_state

                episode_reward += reward  # Dodato: Sabiranje nagrada

            # Podešavanje epsilon i alfa
            epsilon = np.log(e + 2) / (e + 2)
            alpha = np.log(e + 2) / (e + 2)

            # Periodično testiranje trenutne politike
            if (e + 1) % test_interval == 0:
                average_reward = self.test_policy(10)
                rewards_during_learning.append((e + 1, average_reward))
                q_values_during_learning.append(self.get_q_values_snapshot())

            # Dodato: Ispisivanje epizodne nagrade za svaku epizodu
            print(f"Epizoda: {e + 1}, Ukupna nagrada: {episode_reward}")

        return rewards_during_learning, q_values_during_learning

    # Plotovanje V-vrednosti
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

    # Funkcija za pravljenje snapshot-a trenutnih Q-vrednosti u neterminalnim stanjima
    def get_q_values_snapshot(self):
        snapshot = {}
        for state in self.state_space:
            if state not in self.terminal_states:
                snapshot[state] = max(self.Q[state].values())
        return snapshot

    # Funkcija za prikaz prosečnih nagrada tokom učenja
    def plot_rewards_during_learning(self, rewards_during_learning, title):
        episodes, rewards = zip(*rewards_during_learning)
        plt.plot(episodes, rewards, label='Prosečna nagrada')
        plt.xlabel('Epizode')
        plt.ylabel('Prosečna nagrada')
        plt.title(title)
        plt.legend()
        plt.show()

    # Funkcija za prikaz Q-vrednosti tokom učenja
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

    # Funkcija za testiranje trenutne politike kroz određeni broj epizoda
    def test_policy(self, num_episodes):
        total_rewards = 0
        for _ in range(num_episodes):
            state = self.reset()
            done = False
            episode_reward = 0
            while not done:
                action = max(self.Q[state], key=self.Q[state].get)
                next_state, reward, done = self.step(action)
                print(f"Akcija: {action}, Novo stanje: {next_state}, Nagrada: {reward}, Završeno: {done}")
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
env.plot_rewards_during_learning(rewards_during_learning, 'Prosečna nagrada tokom učenja sa promenljivom stopom učenja')
env.plot_q_values_during_learning(q_values_during_learning, 'Q-vrednosti tokom učenja')

# Testiranje sa drugačijim gamma
gamma = 0.999
env.Q = {state: {action: 0 for action in env.actions} for state in env.state_space}  # Reset Q-vrednosti
rewards_during_learning, q_values_during_learning = env.q_learning(num_episodes, alpha, gamma, epsilon, test_interval)
env.plot_rewards_during_learning(rewards_during_learning, 'Prosečna nagrada tokom učenja sa visokim gamma (0.999)')
env.plot_q_values_during_learning(q_values_during_learning, 'Q-vrednosti tokom učenja sa visokim gamma (0.999)')
