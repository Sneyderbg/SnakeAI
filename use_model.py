from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from SnakeEnv import SnakeEnv

import config as cfg

# import time
import pygame
import sys
import numpy as np


def wait_for_key():
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            break


actions = ["<-", "^", "->"]

env = DummyVecEnv([lambda: SnakeEnv(cfg.SHAPE, walls=cfg.USE_WALLS)])

if cfg.MODEL_TYPE == "PPO":
    model = PPO.load(cfg.MODELS_DIR + "/" + cfg.MODEL_FILE_NAME)
elif cfg.MODEL_TYPE == "A2C":
    model = A2C.load(cfg.MODELS_DIR + "/" + cfg.MODEL_FILE_NAME)
else:
    raise TypeError(f"invalid model type in cfg, see {cfg.__file__}")

obs = env.reset()
done = False

np.set_printoptions(precision=2)

# TODO: see https://stable-baselines3.readthedocs.io/en/master/guide/examples.html for evolution

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    next_left, front, right = int(obs[0][0]), int(obs[0][1]), int(obs[0][2])
    food_dir = obs[0][3:6]
    dist_left, dist_front, dist_right = obs[0][6:9]
    game_over = info[0]["game_over"]
    out_str = f"left: {next_left:d}|{dist_left:.2f}, front: {front:d}|{dist_front:.2f}, right: {right:d}|{dist_right:.2f}, food_dir: {
        food_dir} | action: {actions[action[0]]:2} | reward: {reward[0]:3.2f}, done: {done[0]}, game_over: {game_over}"
    print(out_str)
    if done:
        wait_for_key()
        break
    env.render()
    pygame.event.wait()
    # wait_for_key()
