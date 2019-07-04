from collections import deque
from random import random, choice, sample

import gym
import tensorflow as tf
import numpy as np
import keras


class Agent():

    def __init__(self):
        self.num_of_games = 20000

        self.init_exploration = 1
        self.final_exploration = 0.1
        self.final_exploration_frame = 1000000
        self.update_target_model_after_frames = 1000
        self.fit_model_after_frames = 4
        self.xp_length = 100000
        self.batch_size = 32
        self.gamma = 0.99

        self.RAM_SHAPE = (128,)
        self.action_space = np.array([0, 2, 3])
        self.activate_all_actions = np.ones((1, len(self.action_space)))
        self.activate_all_actions_batch = np.ones((self.batch_size, len(self.action_space)))
        self.zero_q_values = np.zeros(len(self.action_space))
        self.action_to_index = {0: 0, 2: 1, 3: 2}

        self.model = self.make_model()
        self.target_model = self.make_model()
        self.target_model.set_weights(self.model.get_weights())  # so they have same weights

        self.xp_memory = deque()

        self.env = gym.make('Breakout-ram-v4')

    def make_model(self):
        ram_input = keras.layers.Input(self.RAM_SHAPE, name='ram_input')
        actions_input = keras.layers.Input(self.action_space.shape, name='actions_input')
        hidden1 = keras.layers.Dense(512, activation='relu', name='hidden1')(ram_input)
        hidden2 = keras.layers.Dense(512, activation='relu', name='hidden2')(hidden1)
        hidden3 = keras.layers.Dense(128, activation='relu', name='hidden3')(hidden2)
        output = keras.layers.Dense(len(self.action_space), name='output')(hidden3)
        filtered_output = keras.layers.multiply([output, actions_input])

        model = keras.models.Model(input=[ram_input, actions_input], output=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')
        print(model.summary())
        return model

    def train(self):
        iter_num = 0
        scores = []
        for game_num in range(self.num_of_games):
            self.env.reset()
            score = 0
            state, reward, terminal, info = self.env.step(1)
            score += reward
            lives_left = info['ale.lives']
            while not terminal:
                action = self.make_action(state, iter_num)
                new_state, reward, terminal, info = self.env.step(action)
                score += reward
                if (info['ale.lives'] < lives_left):  # if life is lost
                    lives_left = info['ale.lives']
                    # reward = -1
                    self.xp_memory.append((state, action, new_state, reward, terminal))
                    new_state, reward, terminal, _ = self.env.step(1)  # start again
                    score += reward
                else:
                    self.xp_memory.append((state, action, new_state, reward, terminal))
                if (len(self.xp_memory) > self.xp_length):
                    self.xp_memory.popleft()
                state = new_state
                iter_num += 1
                if iter_num >= self.batch_size and iter_num % self.fit_model_after_frames == 0:
                    self.fit(sample(self.xp_memory, self.batch_size))
                if iter_num % self.update_target_model_after_frames == 0:
                    self.target_model.set_weights(self.model.get_weights())
            scores.append(score)
            if game_num % 100 == 0:
                print(f"Game number: {game_num}\t\tAvg score:{np.mean(scores)}\t\tFrame:{iter_num}")
                scores = []
                self.model.save(f"models/model_game_{game_num}.mdl")
            if game_num == self.num_of_games - 1:
                self.model.save(f"models/final_model.mdl")

    def fit(self, batch):

        states = np.zeros((len(batch), self.RAM_SHAPE[0]), dtype=np.uint8)
        actions_encoded = np.zeros((len(batch), len(self.action_space)), dtype=np.uint8)
        new_states = np.zeros((len(batch), self.RAM_SHAPE[0]), dtype=np.uint8)
        rewards = np.zeros((len(batch),), dtype=np.float32)
        terminals = np.zeros((len(batch),), dtype=np.bool)

        # copying by value
        for i in range(len(batch)):
            states[i] = batch[i][0]
            actions_encoded[i][self.action_to_index[batch[i][1]]] = 1
            new_states[i] = batch[i][2]
            rewards[i] = batch[i][3]
            terminals[i] = batch[i][4]

        next_Q_values = self.target_model.predict([new_states, self.activate_all_actions_batch])
        next_Q_values[terminals] = self.zero_q_values
        #  Q(s,a) = r + gamma*max(Q(s',a'))
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)

        self.model.fit([states, actions_encoded], actions_encoded * np.expand_dims(Q_values, axis=1), nb_epoch=1,
                       batch_size=self.batch_size, verbose=0)

    def make_action(self, state, iter_num):
        epsilon = self.final_exploration
        if iter_num < self.final_exploration_frame:
            epsilon = self.init_exploration - (iter_num / self.final_exploration_frame) * (
                    self.init_exploration - self.final_exploration)
        if random() < epsilon:
            return choice(self.action_space)
        else:
            return self.action_space[
                np.argmax(self.model.predict([np.expand_dims(state, axis=0), self.activate_all_actions]))]


if __name__ == '__main__':
    a = Agent()
    a.train()
