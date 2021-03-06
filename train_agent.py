from collections import deque
from random import random, choice, sample

import gym
import tensorflow as tf
import numpy as np
import keras
import sys
import argparse
import cv2


class Agent():

    def __init__(self, path=None, iter_num=0, conv=False, best_score=0.0):
        self.num_of_games = 100000
        self.iter_num = iter_num
        self.best_score = best_score
        self.conv = conv

        self.init_exploration = 1
        self.final_exploration = 0.1
        self.final_exploration_frame = 1000000
        self.update_target_model_after_frames = 1000
        self.fit_model_after_frames = 4
        self.xp_length = 100000
        self.batch_size = 32
        self.gamma = 0.99

        self.RMS_learning_rate = 0.00025
        self.RMS_rho = 0.95
        self.RMS_epsilon = 0.01

        self.RAM_SHAPE = (128,)
        self.PP_FRAME_SHAPE = (84, 84, 1)
        self.action_space = np.array([0, 2, 3])
        self.activate_all_actions = np.ones((1, len(self.action_space)))
        self.activate_all_actions_batch = np.ones((self.batch_size, len(self.action_space)))
        self.zero_q_values = np.zeros(len(self.action_space))
        self.action_to_index = {0: 0, 2: 1, 3: 2}

        self.xp_memory = deque()

        if path is not None:
            self.model = keras.models.load_model(path)
            self.target_model = keras.models.load_model(path)
            if not conv:
                self.env = gym.make('Breakout-ramDeterministic-v4')  # always skip 4 frames and no randomness
            else:
                self.env = gym.make('BreakoutDeterministic-v4')
        else:
            if not conv:
                self.model = self.make_model()
                self.target_model = self.make_model()
                self.target_model.set_weights(self.model.get_weights())  # so they have same weights
                self.env = gym.make('Breakout-ramDeterministic-v4')  # always skip 4 frames and no randomness
            else:
                self.model = self.make_conv_model()
                self.target_model = self.make_conv_model()
                self.target_model.set_weights(self.model.get_weights())  # so they have same weights
                self.env = gym.make('BreakoutDeterministic-v4')

    def make_model(self):
        ram_input = keras.layers.Input(self.RAM_SHAPE, name='ram_input')
        actions_input = keras.layers.Input(self.action_space.shape, name='actions_input')
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(ram_input)
        hidden1 = keras.layers.Dense(512, activation='relu', name='hidden1')(normalized)
        dropout = keras.layers.Dropout(0.2)(hidden1)
        hidden3 = keras.layers.Dense(128, activation='relu', name='hidden3')(dropout)
        output = keras.layers.Dense(len(self.action_space), name='output')(hidden3)
        filtered_output = keras.layers.multiply([output, actions_input])

        model = keras.models.Model(input=[ram_input, actions_input], output=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=self.RMS_learning_rate, rho=self.RMS_rho, epsilon=self.RMS_epsilon)
        model.compile(optimizer, loss='mse')
        print(model.summary())
        return model

    def make_conv_model(self):
        frames_input = keras.layers.Input(self.PP_FRAME_SHAPE, name='frames_input')
        actions_input = keras.layers.Input(self.action_space.shape, name='actions_input')
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

        conv_1 = keras.layers.convolutional.Convolution2D(
            16, 8, 8, input_shape=(1, 84, 84), subsample=(4, 4), activation='relu'
        )(normalized)

        conv_2 = keras.layers.convolutional.Convolution2D(
            32, 4, 4, input_shape=(20, 20, 16), subsample=(2, 2), activation='relu'
        )(conv_1)
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)

        output = keras.layers.Dense(len(self.action_space))(hidden)

        filtered_output = keras.layers.multiply([output, actions_input])

        model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=self.RMS_learning_rate, rho=self.RMS_rho, epsilon=self.RMS_epsilon)
        model.compile(optimizer, loss='mse')

        print(model.summary())
        return model

    def train(self):
        scores = []
        for game_num in range(self.num_of_games):
            self.env.reset()
            score = 0
            state, reward, terminal, info = self.env.step(1)
            if self.conv:
                state = self.preprocess(state)
            score += reward
            lives_left = info['ale.lives']
            while not terminal:
                action = self.make_action(state, self.iter_num)
                new_state, reward, terminal, info = self.env.step(action)
                if self.conv:
                    new_state = self.preprocess(new_state)
                score += reward
                if (info['ale.lives'] < lives_left):  # if life is lost
                    lives_left = info['ale.lives']
                    reward = -3
                    self.xp_memory.append((state, action, new_state, reward, terminal))
                    new_state, reward, terminal, _ = self.env.step(1)  # start again
                    if self.conv:
                        new_state = self.preprocess(new_state)
                    score += reward
                else:
                    self.xp_memory.append((state, action, new_state, reward, terminal))
                if (len(self.xp_memory) > self.xp_length):
                    self.xp_memory.popleft()
                state = new_state
                self.iter_num += 1
                if self.iter_num % self.fit_model_after_frames == 0:
                    self.fit(sample(self.xp_memory, min(self.batch_size, len(self.xp_memory))))
                if self.iter_num % self.update_target_model_after_frames == 0:
                    self.target_model.set_weights(self.model.get_weights())
            scores.append(score)
            if score >= self.best_score:
                self.best_score = score
                print(f"New best score: {self.best_score}")
                self.model.save(f"best_models/best_model_score_{int(self.best_score)}")
            if game_num % 100 == 0:
                print(f"Game number: {game_num}\t\tAvg score:{np.mean(scores)}\t\tFrame:{self.iter_num}")
                scores = []
                self.model.save(f"models/model_game_{game_num}.mdl")
            if game_num == self.num_of_games - 1:
                self.model.save(f"models/final_model.mdl")

    def fit(self, batch):
        if self.conv:
            states = np.zeros((len(batch), self.PP_FRAME_SHAPE[0], self.PP_FRAME_SHAPE[1], self.PP_FRAME_SHAPE[2]),
                              dtype=np.uint8)

            new_states = np.zeros((len(batch), self.PP_FRAME_SHAPE[0], self.PP_FRAME_SHAPE[1], self.PP_FRAME_SHAPE[2]),
                                  dtype=np.uint8)
        else:
            states = np.zeros((len(batch), self.RAM_SHAPE[0]), dtype=np.uint8)
            new_states = np.zeros((len(batch), self.RAM_SHAPE[0]), dtype=np.uint8)

        actions_encoded = np.zeros((len(batch), len(self.action_space)), dtype=np.uint8)
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

    # taken from http://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
    def preprocess(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[17:101, :]
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(observation, (84, 84, 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--conv", action="store_true",
                        help="use convolutional network on frame inputs")
    parser.add_argument("-m", "--model",
                        help="continue traning of a model")
    args = parser.parse_args()

    if args.model:
        a = Agent(conv=args.conv, path=args.model, iter_num=1000000, best_score=350.0)
        a.train()
    else:
        a = Agent(conv=args.conv)
        a.train()
