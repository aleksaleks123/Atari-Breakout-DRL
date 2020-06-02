import sys
from time import sleep
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import numpy as np
import keras
import argparse
from random import random, choice
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

    def run(self, num_of_games, sleep_time, render, epsilon):
        print("Running ...")
        scores =[]
        for game_num in range(num_of_games):
            self.env.reset()
            score = 0
            state, reward, terminal, info = self.env.step(1)
            if self.conv:
                state = self.preprocess(state)
            score += reward
            lives_left = info['ale.lives']
            while not terminal:
                action = self.make_action(state, epsilon)
                state, reward, terminal, info = self.env.step(action)
                if self.conv:
                    state = self.preprocess(state)
                score += reward
                if render:
                    self.env.render()
                sleep(sleep_time/1000)
                if (info['ale.lives'] < lives_left):  # if life is lost
                    lives_left = info['ale.lives']
                    state, reward, terminal, _ = self.env.step(1)  # start again
                    if self.conv:
                        state = self.preprocess(state)
                    score += reward
            scores.append(score)
        print(f'Average score: {np.mean(scores)}')
        print(f'All scores: {scores}')

    def make_action(self, state, epsilon):
        if random() < epsilon:
            return choice(self.action_space)
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

    parser.add_argument("model",
                        help="Path to model that is being tested")

    parser.add_argument("-r", "--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("-c", "--conv", action="store_true",
                        help="Model is convolutional")
    parser.add_argument("-g", "--games",
                        help="Number of games", type=int, default=3)
    parser.add_argument("-s", "--sleep",
                        help="Sleep time in milliseconds between actions", type=float, default=10)
    parser.add_argument("-e", "--epsilon",
                        help="Epsilon coefficient, that determines the percentage of random actions", type=float, default=0)
    args = parser.parse_args()

    at = AgentTest(args.model, conv=args.conv)
    at.run(args.games, args.sleep, args.render, args.epsilon)
