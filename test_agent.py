import sys
from time import sleep

import gym
import tensorflow as tf
import numpy as np
import keras
import argparse
import cv2

class AgentTest():
    def __init__(self, file, conv):
        self.action_space = np.array([0, 2, 3])
        self.activate_all_actions = np.ones((1, len(self.action_space)))
        self.model = keras.models.load_model(file)
        if not conv:
            self.env = gym.make('Breakout-ramDeterministic-v4')  # always skip 4 frames and no randomness
        else:
            self.env = gym.make('BreakoutDeterministic-v4')
        self.conv = conv

    def run(self, num_of_games):
        for game_num in range(num_of_games):
            self.env.reset()
            state, _, terminal, info = self.env.step(1)
            if self.conv:
                state = self.preprocess(state)
            lives_left = info['ale.lives']
            while not terminal:
                action = self.make_action(state)
                state, _, terminal, info = self.env.step(action)
                if self.conv:
                    state = self.preprocess(state)
                self.env.render()
                sleep(0.01)
                if (info['ale.lives'] < lives_left):  # if life is lost
                    lives_left = info['ale.lives']
                    state, _, terminal, _ = self.env.step(1)  # start again
                    if self.conv:
                        state = self.preprocess(state)

    def make_action(self, state):
        return self.action_space[
            np.argmax(self.model.predict([np.expand_dims(state, axis=0), self.activate_all_actions]))]

    # taken from http://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
    def preprocess(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110, :]
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(observation, (84, 84, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("model",
                        help="Path to model that is being tested")

    parser.add_argument("-c", "--conv", action="store_true",
                        help="Model is convolutional")
    parser.add_argument("-g", "--games",
                        help="Number of games", type=int)
    args = parser.parse_args()

    if args.games:
        num_of_games = args.games
    else:
        num_of_games = 3

    args = parser.parse_args()
    at = AgentTest(args.model, conv=args.conv)
    at.run(num_of_games)
