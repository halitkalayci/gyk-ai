import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os

# 1. Ortam Ayarları
grid_size = 5
actions = ['up', 'down', 'left', 'right']

# Ödüller
reward_goal = 1
reward_obstacle = -1
reward_step = -0.01

start_state = (0, 0)
goal_state = (4, 4)

q_table = np.zeros((grid_size, grid_size, len(actions)))

# 2. Hiperparametreler
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 500

# 3. Hareket Fonksiyonu
def take_action(state, action):
    x, y = state
    if action == 'up':
        x = max(0, x - 1)
    elif action == 'down':
        x = min(grid_size - 1, x + 1)
    elif action == 'left':
        y = max(0, y - 1)
    elif action == 'right':
        y = min(grid_size - 1, y + 1)
    return (x, y)

# 4. Eğitim
rewards_per_episode = []

for episode in range(num_episodes):
    state = start_state
    total_reward = 0

    for step in range(100):
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, len(actions) - 1)
        else:
            action_index = np.argmax(q_table[state[0], state[1]])

        action = actions[action_index]
        new_state = take_action(state, action)

        if new_state == goal_state:
            reward = reward_goal
        elif new_state == state:
            reward = reward_obstacle
        else:
            reward = reward_step

        old_q_value = q_table[state[0], state[1], action_index]
        next_max = np.max(q_table[new_state[0], new_state[1]])

        new_q_value = old_q_value + learning_rate * (reward + discount_factor * next_max - old_q_value)
        q_table[state[0], state[1], action_index] = new_q_value

        state = new_state
        total_reward += reward

        if state == goal_state:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    rewards_per_episode.append(total_reward)

# 5. Eğitim Sonrası Ödül Grafiği
plt.plot(rewards_per_episode)
plt.xlabel('Bölüm')
plt.ylabel('Toplam Ödül')
plt.title('Eğitim Süreci - Toplam Ödül')
plt.show()

# 6. Oyun Animasyonu (Ajani İzleme)
def render(state):
    os.system('cls' if os.name == 'nt' else 'clear')  # Terminali temizle
    for i in range(grid_size):
        row = ''
        for j in range(grid_size):
            if (i, j) == state:
                row += ' A '  # Agent
            elif (i, j) == goal_state:
                row += ' G '  # Goal
            else:
                row += ' . '  # Boş
        print(row)
    print("\n")

# 7. Ajanın Oynatılması
state = start_state
steps = 0
print("Ajan başlıyor...\n")
time.sleep(2)

while state != goal_state and steps < 50:
    render(state)
    action_index = np.argmax(q_table[state[0], state[1]])
    action = actions[action_index]
    state = take_action(state, action)
    steps += 1
    time.sleep(0.5)

render(state)
print(f"Ajan {steps} adımda hedefe ulaştı!")
