from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from matplotlib import pyplot as plt
from datetime import datetime

from SnakeEnv import SnakeEnv
import config as cfg


env = SnakeEnv(cfg.SHAPE, render=False, walls=cfg.USE_WALLS)
env = make_vec_env(lambda: env, monitor_dir=cfg.MONITOR_DIR) # type: ignore

custom_hyperparams = {"learning_rate": 0.0001, "tensorboard_log": cfg.TENSOR_LOG_DIR}

if cfg.MODEL_TYPE == "PPO":
    model = PPO.load(cfg.MODELS_DIR + "/" + cfg.MODEL_FILE_NAME, custom_objects=custom_hyperparams, device='cuda')
elif cfg.MODEL_TYPE == "A2C":
    model = A2C.load(cfg.MODELS_DIR + "/" + cfg.MODEL_FILE_NAME, custom_objects=custom_hyperparams, device='cuda')
else:
    raise TypeError(f"invalid model type in cfg, see {cfg.__file__}")

model.set_env(env)
model.learn(total_timesteps=40000, progress_bar=True)

# title = f"lr:{model.learning_rate} g:{model.gamma} entr:{model.ent_coef}"

# results_plotter.plot_results(
#     [cfg.MONITOR_DIR], int(10e6), results_plotter.X_TIMESTEPS, title)

# date_time = datetime.now()
# plot_save_file = cfg.RESULTS_DIR + \
#     f"/{date_time.year}-{date_time.month}-{date_time.day}_{date_time.hour}-{date_time.minute}-{date_time.second}.png"

# plt.savefig(plot_save_file)
# print(f"plot saved as {plot_save_file}")
# plt.show()
model.save(cfg.MODELS_DIR + "/" + cfg.MODEL_FILE_NAME)