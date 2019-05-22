import gym
import random
import numpy as np
from models import neural_network_model

env = gym.make('CartPole-v1')
env._max_episode_steps = 500
env.reset()
score_requirement = 50
initial_games = 10000

model = neural_network_model(env.observation_space.shape[0])
model.load('openaigym-cartpole-200')

scores = []
choices = []

for each_game in range(1):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    counter = 0
    while True:
        counter += 1;
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)

        if len(prev_obs) > 0:
            game_memory.append([prev_obs, action])

        print('Step', str(counter), 'pos:', new_observation[0])
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break

    scores.append(score)

print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices),
      choices.count(0)/len(choices)))