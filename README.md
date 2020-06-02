# Atari Breakout - Deep Reinforcement Learning

Python application for training models to play Atari Breakout and testing their performances.

Used enviroment: OpenAI Gym

# How to Use
### Train
```python train_agent.py [-h] [-c] [-m MODEL]```

-h => show help

-c => to use game frames as input and convolutional network to train

-m MODEL => to continue training on existing model (MODEL is the path to it)

### Test
```python test_agent.py [-h] [-c] [-g GAMES] [-s SLEEP] model```

-h => show help

-r => render the environment, in order for the gameplay to be seen

-c => to use game frames as input and test convolutive network model

-g GAMES => number of games to play (default is 3)

-s SLEEP => sleep time in milliseconds between actions (default is 10)

-e EPSILON => Epsilon coefficient, that determines the percentage of random actions (default is 0)

model => path to model we wish to test
