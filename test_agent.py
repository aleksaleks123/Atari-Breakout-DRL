import sys
from time import sleep

import gym
import tensorflow as tf
import numpy as np
import keras


class AgentTest():
    def __init__(self, file):
        self.action_space = np.array([0, 2, 3])
        self.activate_all_actions = np.ones((1, len(self.action_space)))
        self.model = keras.models.load_model(file)
        self.env = gym.make('Breakout-ram-v4')

    def run(self, num_of_games):
        for game_num in range(num_of_games):
            self.env.reset()
            state, _, terminal, info = self.env.step(1)
            lives_left = info['ale.lives']
            while not terminal:
                action = self.make_action(state)
                state, _, terminal, info = self.env.step(action)
                self.env.render()
                sleep(0.01)
                if (info['ale.lives'] < lives_left):  # if life is lost
                    lives_left = info['ale.lives']
                    state, _, terminal, _ = self.env.step(1)  # start again

    def make_action(self, state):
        return self.action_space[
            np.argmax(self.model.predict([np.expand_dims(state, axis=0), self.activate_all_actions]))]


if __name__ == '__main__':
    at = AgentTest(sys.argv[1])
    at.run(3)
