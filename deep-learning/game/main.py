import gym
import torch.nn as nn # Yapay sinir ağı bileşenleri.
import torch
from collections import deque # double-ended queue
import torch.optim as optim # Optimizasyon algoritmaları.

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Ağ oluştur.
policy_net = QNetwork(state_size, action_size).to(device) # aktif öğrenen ağ
target_net = QNetwork(state_size, action_size).to(device) # sadece sabit hedef veren ağ

# en başta ağları eşitle.
target_net.load_state_dict(policy_net.state_dict())

# Optimizer (Adam)

# fc1  -> Ağırlıklar (W1) -> bias (b1)
# fc2  -> Ağırlıklar (W2) -> bias (b2)
# fc3  -> Ağırlıklar (W3) -> bias (b3)

#Ağırlıkları tahminleri daha doğru yapacak şekilde adım adım günceller.

# Adaptive Moment Estimation
# import torch.optim as optim
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
# LR => Ağırlıkları ne kadar hızlı değiştireceğiz?
# Az değiştir -> Yavaş öğrenir ama sağlam olur.
# Çok değiştir -> Hızlı öğrenir ama bazı durumlarda saçmalayabilir.

import random
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size) # rastgele bir aksiyon çevir geriye.
    else:
        # Durumu tensöre çevir. Ağdan tahmin al. (Öğrenilen bir işlem yap.)
        # (batch_size, özellik_sayısı) -> (4,) -> (1,4)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return policy_net(state).argmax().item()
        
# Her bir bölüm için oyunu yeniden başlat, bitene kadar select_action fonk. ile rastgele ya da öğrenilmiş hareket yap.
epsilon = EPS_START

for episode in range(EPISODES):
    state, _ = env.reset() #Ortamı sıfırla
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)

        # İlgili seçilen eylemi uygula ve yeni durumdaki parametreleri al.
        next_state, reward, terminated, truncated, _ = env.step(action)

        #Bitiş durumu, bi şekilde oyun kesildiyse (kapandı ya da yandık.)
        done = truncated or terminated

        memory.append((state,action,reward,next_state,done))

        state = next_state
        total_reward += reward
    
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    # Epsilon = Rastgelelik oranı

    # Belirli aralıklara ağı güncelle.
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode}: Total Reward: {total_reward} Epsilon: {epsilon}")
