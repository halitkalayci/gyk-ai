import gym
import torch.nn as nn # Yapay sinir ağı bileşenleri.
import torch
from collections import deque # double-ended queue

env = gym.make('CartPole-v1', render_mode="human")

# Hiperparametreler
EPISODES = 50
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000 # kaç adet deney (experience)?
EPS_START = 1.0 # Epsilon=Keşif oranı
EPS_END = 0.01 # Minimum epsilon 0.01
EPS_DECAY = 0.995 # Epsilonun her adımda çarpılarak azaltılma oranı.
TARGET_UPDATE = 10 # Hedef ağı güncelleme sıklığı (episode bazında)
#

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size): # Bu sinir ağının katmanlarını tanımla.
        super(QNetwork, self).__init__()  # Pytorch'a bu class için bir NN oluşturmasını söyle.
        self.fc1 = nn.Linear(state_size, 24) # Durumu al 24 sayıya dönüştür. # İlk katman fully connected layer.
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, action_size) # 48->2 => Sol => 1.5 Sağ => 2.0
    
    def forward(self, x):
        # İleri besleme adımı, ReLu aktivasyon fonksiyonu.
        x = torch.relu(self.fc1(x)) # 1.katmana girmiş ve non-linearlik için akt. fonksiyonuna uğratılmış.
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
memory = deque(maxlen=MEMORY_SIZE)
