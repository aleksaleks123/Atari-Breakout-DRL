import gym
from time import sleep

env = gym.make('Breakout-ram-v4')
print(env.unwrapped.get_action_meanings())
for game_num in range(100):
    env.reset()
    score = 0
    state, _, terminal, info = env.step(1)
    lives_left = info['ale.lives']
    while not terminal:
        action = 3
        state, reward, terminal, info = env.step(action)
        score += reward
        env.render()
        sleep(0.01)
        print(score)
        if (info['ale.lives'] < lives_left):  # if life is lost
            lives_left = info['ale.lives']
            state, reward, terminal, i = env.step(1)  # start again
            score += reward

#
# for i_episode in range(20):
#     observation = env.reset()
#     env.step(1)
#     for t in range(1000):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(1)
#         if(info['ale.lives'] == 0):
#             print(reward)
#             print(done)
#             reward = -3
#         sleep(0.1)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# observation = env.reset()
# print(type(observation))
#
# env.step(1)
# print(observation)
# env.step(1)
# env.render()
# sleep(2)
# observation, reward, done, info = env.step(3)
# env.render()
# print(observation.shape)
# sleep(10)

env.close()
