from stable_baselines3 import A2C, PPO # type: ignore
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from matplotlib import pyplot as plt
from datetime import datetime

from SnakeEnv import SnakeEnv
import config as cfg

env = SnakeEnv(cfg.SHAPE, render=False, walls=cfg.USE_WALLS)
env = make_vec_env(lambda: env, monitor_dir=cfg.MONITOR_DIR)  # type: ignore

lr = 0.001

eval_callback = EvalCallback(env, n_eval_episodes=5, eval_freq=5000, best_model_save_path=cfg.MODELS_DIR, render=False)

if cfg.MODEL_TYPE == "PPO":
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=lr, tensorboard_log=cfg.TENSOR_LOG_DIR, policy_kwargs= {"net_arch":[32, 32]}, device='cuda')
elif cfg.MODEL_TYPE == "A2C":
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=lr, tensorboard_log=cfg.TENSOR_LOG_DIR, policy_kwargs= {"net_arch":[32, 32]}, device='cuda')
else:
    raise TypeError(f"invalid model type in cfg, see {cfg.__file__}")

model = model.learn(total_timesteps=20000, callback=eval_callback, progress_bar=True)

# title = f"lr:{model.learning_rate} g:{model.gamma} entr:{model.ent_coef}"

# results_plotter.plot_results(
#     [cfg.MONITOR_DIR], int(10e6), results_plotter.X_TIMESTEPS, title)

# date_time = datetime.now()
# plot_save_file = cfg.RESULTS_DIR + \
#     f"/{date_time.year}-{date_time.month}-{date_time.day}_{date_time.hour}-{date_time.minute}-{date_time.second}.png"

# # plt.savefig(plot_save_file)
# print(f"plot saved as {plot_save_file}")
# plt.show()
model.save(cfg.MODELS_DIR + "/" + cfg.MODEL_FILE_NAME)