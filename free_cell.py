from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import free_cell_game as fc

class MyAgent:
    def __init__(self, state_size):
        self.state_size = state_size

        self.replay_buffer = deque(maxlen=50000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10

        self.main_network = self.get_nn()
        self.target_network = self.get_nn()

        self.target_network.set_weights(self.main_network.get_weights())

    def get_nn(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def save_experience(self, state, action, reward, next_state, terminal, available_actions):
        self.replay_buffer.append((state, action, reward, next_state, terminal, available_actions))

    def train_main_network(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)

        for state, act, reward, next_state, done, available_actions in minibatch:
            target = reward
            action_encoding = np.zeros((len(available_actions), 51))
            for j, act in enumerate(available_actions):
                from_index = ord(act['from'][0]) - ord('A')
                from_value = int(act['from'][1:])
                dest_index = ord(act['dest'][0]) - ord('A')
                dest_value = int(act['dest'][1:])
                is_move_to_foundation = int(act['isMoveToFoundation'])

                action_encoding[j, from_index] = from_value
                action_encoding[j, dest_index + 13] = dest_value
                action_encoding[j, 26] = is_move_to_foundation

            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                state_embedding = self.target_network.predict(next_state)
                state_embedding_repeated = np.repeat(state_embedding, len(available_actions), axis=0)
                merged = Concatenate(axis=1)([state_embedding_repeated, action_encoding])
                next_act_values = self.target_network.predict(merged)
                max_action_idx = np.argmax(next_act_values)
                target = reward + self.gamma * next_act_values[max_action_idx]

            state = np.reshape(state, [1, self.state_size])
            state_embedding = self.main_network.predict(state)
            state_embedding_repeated = np.repeat(state_embedding, len(available_actions), axis=0)
            merged = Concatenate(axis=1)([state_embedding_repeated, action_encoding])

            act_values = self.main_network.predict(merged)
            max_action_idx = np.argmax(act_values)
            act_values[max_action_idx] = target

            self.main_network.fit(merged, act_values, epochs=1, verbose=0)

    def make_decision(self, state, available_actions):
        if random.uniform(0,1) < self.epsilon:
            return np.random.choice(available_actions)
        
        self.action_size = len(available_actions)
        state = np.reshape(state, [1, self.state_size])

        action_encoding = np.zeros((len(available_actions), 51))
        for i, action in enumerate(available_actions):
            from_index = ord(action['from'][0]) - ord('A')
            from_value = int(action['from'][1:])
            dest_index = ord(action['dest'][0]) - ord('A')
            dest_value = int(action['dest'][1:])
            is_move_to_foundation = int(action['isMoveToFoundation'])

            action_encoding[i, from_index] = from_value
            action_encoding[i, dest_index + 13] = dest_value
            action_encoding[i, 26] = is_move_to_foundation

        state_embedding = self.main_network.predict(state)
        state_embedding_repeated = np.repeat(state_embedding, self.action_size, axis=0)
        merged = Concatenate(axis=1)([state_embedding_repeated, action_encoding])

        act_values = self.main_network.predict(merged)
        print(act_values, available_actions)
        max_action_idx = np.argmax(act_values)
        max_action = available_actions[max_action_idx]
        return max_action



# Khởi tạo môi trường
env = fc.FreecellEnvironment(maxMove=150)
state = env.reset()

# Định nghĩa state_size và action_size
state_size = len(env.getState())

# Định nghĩa tham số khác
n_episodes = 200
n_timesteps = 300
batch_size = 64

# Khởi tạo agent
my_agent = MyAgent(state_size)
total_time_step = 0

for ep in range(n_episodes):
    ep_rewards = 0
    state = env.reset()

    for t in range(n_timesteps):

        total_time_step += 1
        # Cập nhật lại target NN mỗi my_agent.update_targetnn_rate
        if total_time_step % my_agent.update_targetnn_rate == 0:
            # Có thể chọn cách khác = weight của targetnetwork = 0 * weight của targetnetwork  + 1  * weight của mainnetwork
            my_agent.target_network.set_weights(my_agent.main_network.get_weights())
        
        movables = env.get_movable()

        action = my_agent.make_decision(state, movables)

        next_state, reward, terminal, _,= env.step(action)
        my_agent.save_experience(state,action, reward, next_state, terminal, movables)

        state = next_state
        ep_rewards += reward
        # print(action, reward, "numOfMoves: ", env.numOfMove)
        env.printBoard()
        if terminal:
            print("Ep ", ep+1, " reach terminal with reward = ", ep_rewards)
            break

        if len(my_agent.replay_buffer) > batch_size:
            my_agent.train_main_network(batch_size)

    if ep+1%10==0:
        my_agent.main_network.save("train_agent_"+str(ep+1)+".h5")

    if my_agent.epsilon > my_agent.epsilon_min:
        my_agent.epsilon = my_agent.epsilon * my_agent.epsilon_decay

# Save weights
my_agent.main_network.save("train_agent_final.h5")