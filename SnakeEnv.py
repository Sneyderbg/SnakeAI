import gymnasium as gym
from gymnasium import spaces
from BaseSnake import SnakeGame
import numpy as np
from PIL import Image, ImageDraw
from math import acos
import config as cfg

class SnakeEnv(gym.Env):
    LEFT = -1
    FORWARD = 0
    RIGHT = 1

    def __init__(self, shape, render=True, render_mode="pygame", walls=False) -> None:
        super(SnakeEnv, self).__init__()
        self.render_mode = render_mode
        self.BaseSnake = SnakeGame(shape, limit_collision=walls)
        self.BaseSnake.fps = cfg.STEPS_PER_SEC
        if render:
            self.BaseSnake.start(loop=False)

        self.action_space = spaces.Discrete(3)
        # self.observation_space = spaces.Box(low=0, high=4, shape=(shape[0] * shape[1],), dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)

        self.low, self.high = self.LEFT, self.RIGHT
        self.steps_without_eating = 0
        self.reward = 0
        self.prev_shaping = None

    def calculate_grid(self):
        # 0 = nothing
        # 1 = head
        # 2 = body
        # 3 = tail
        # 4 = food

        shape = (self.BaseSnake.WIDTH, self.BaseSnake.HEIGHT)
        snake = self.BaseSnake.snake
        food = self.BaseSnake.food
        grid = np.zeros(shape, dtype=np.int32)
        for i, (x, y) in enumerate(snake):
            x, y = int(x), int(y)
            if i == 0:
                grid[x][y] = 1
            elif i == len(snake) - 1:
                grid[x][y] = 3
            else:
                grid[x][y] = 2

        grid[food.x, food.y] = 4
        return np.ndarray.flatten(grid)

    def calculate_obs(self):
        snake = self.BaseSnake.snake
        head = snake[0]
        food = self.BaseSnake.food
        shape = (self.BaseSnake.WIDTH, self.BaseSnake.HEIGHT)

        # obstacles
        next_left = head + self.BaseSnake._get_turn_dir("left")
        next_right = head + self.BaseSnake._get_turn_dir("right")
        next_front = head + self.BaseSnake.snake_dir

        obstacles = [next_left, next_front, next_right]
        obstacles = list(
            map(
                lambda new_head: self.BaseSnake.get_collision(new_head)
                in ["wall", "snake"],
                obstacles,
            )
        )

        turn_left_dir = self.BaseSnake._get_turn_dir("left")
        turn_right_dir = self.BaseSnake._get_turn_dir("right")
        dont_turn_dir = self.BaseSnake.snake_dir

        # distance to obstacles
        dist_left = self.BaseSnake.get_collision_distance(turn_left_dir)
        dist_right = self.BaseSnake.get_collision_distance(turn_right_dir)
        dist_front = self.BaseSnake.get_collision_distance(dont_turn_dir)

        # scale distances
        dist_left = dist_left / self.BaseSnake.get_related_max(turn_left_dir)
        dist_right = dist_right / self.BaseSnake.get_related_max(turn_right_dir)
        dist_front = dist_front / self.BaseSnake.get_related_max(dont_turn_dir)

        food_dir = food - head
        snake_dir = self.BaseSnake.snake_dir

        # normalize
        food_dir = food_dir.normalize()
        snake_dir = snake_dir.normalize()

        # dot product (compare the 2 vectors)
        # dot_norm = food_dir.dot(snake_dir)
        angle = snake_dir.angle_to(food_dir)
        food_right = 0
        food_left = 0
        food_forward = 0
        if abs(angle) < 180:
            if angle < 0:
                food_right = 1
            else:
                food_left = 1
        elif abs(angle) > 180:
            if angle < 0:
                food_left = 1
            else:
                food_right = 1
        else:
            food_forward = 1

        obs = (
            obstacles
            + [food_left, food_forward, food_right]
            + [dist_left, dist_front, dist_right]
        )

        return np.array(obs, dtype=np.float32)

    def calculate_reward(self) -> float:
        if self.BaseSnake.game_over:
            return -100
        if self.steps_without_eating > 100:
            return -100

        food = self.BaseSnake.food
        head = self.BaseSnake.snake[0]

        reward = 0
        shaping = -head.distance_to(food) + 10 * (len(self.BaseSnake.snake) - 2)
        # shaping = -head.distance_to(food) + head.distance_to(self.BaseSnake.snake[-1])*.1 + 10 * (len(self.BaseSnake.snake) - 2)

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        return reward

        # diference = self.euclidean(
        #     head) - self.euclidean(self.BaseSnake.last_pos)

        # # if (diference > 0):
        # #     print("              ALEJANDO")
        # # else:
        # #     print("              ACERCANDO")

        # # self.reward = - (diference if diference <= 0 else 0.9 * diference) # more punishment to bad actions
        # self.reward += (10 if self.BaseSnake.ate_last_frame else -1.1)
        # print(self.reward)

        # return self.reward

    def euclidean(self, pos):
        food = self.BaseSnake.food
        head = pos
        dist_scaled = food.distance_to(head) / self.BaseSnake._shape.magnitude()
        return dist_scaled

    def manhattan(self, pos):
        x, y = pos
        xf, yf = self.BaseSnake.food
        width, height = self.BaseSnake._shape
        manhattan = (abs(xf - x) + abs(yf - y)) / (width + height - 2)
        return manhattan

    def start(self):
        self.BaseSnake.start(loop=False)

    def close(self):
        self.BaseSnake.close()

    def reset(self, seed=None):
        self.BaseSnake.reset()
        self.steps_without_eating = 0
        self.reward = 0
        self.prev_shaping = None
        # obs = self.calculate_grid()
        obs = self.calculate_obs()
        return obs, {}

    def rescale_action(self, action):
        return (action / 2) * (self.high - self.low) + self.low

    def step(self, action):
        action = self.rescale_action(action)
        if action == self.LEFT:
            self.BaseSnake.turn_dir("left")
        elif action == self.RIGHT:
            self.BaseSnake.turn_dir("right")
        elif action == self.FORWARD:
            pass
        else:
            raise ValueError(
                "Received invalid action={} which is not part of the action space".format(
                    action
                )
            )

        self.steps_without_eating += 1
        self.BaseSnake.update(is_step=True)
        if self.BaseSnake.ate_last_frame:
            self.steps_without_eating = 0

        # obs = self.calculate_grid()
        obs = self.calculate_obs()
        reward = self.calculate_reward()
        done = len(self.BaseSnake.snake) == (
            self.BaseSnake.WIDTH * self.BaseSnake.HEIGHT - 1
        )
        truncated = self.BaseSnake.game_over or reward <= -50
        info = {"game_over": self.BaseSnake.game_over}

        return obs, reward, done, truncated, info

    def generate_ascii(self):
        ascii_mapping = {
            "head": {
                "right": ">",
                "down": "v",
                "left": "<",
                "up": "^",
            },
            "body": "#",
            "food": "*",
        }
        width, height = self.BaseSnake.WIDTH, self.BaseSnake.HEIGHT
        grid = ["░" for _ in range(width)]
        grid.append("\n")
        grid = grid * height
        for i, (x, y) in enumerate(self.BaseSnake.snake):
            pos = (width + 1) * int(y) + int(x)
            if i == 0:
                c = "o"
                for ki, v in enumerate(self.BaseSnake.DIRECTIONS.values()):
                    if self.BaseSnake.snake_dir == v:
                        c = list(ascii_mapping["head"].values())[ki]
                grid[pos] = c
            else:
                grid[pos] = ascii_mapping["body"]

        x, y = self.BaseSnake.food
        pos = (width + 1) * int(y) + int(x)
        grid[pos] = ascii_mapping["food"]
        return "".join(grid)

    def generate_image(self):
        sqr_size = self.BaseSnake.sqr_size
        w, h = self.BaseSnake._shape
        w, h = int(w), int(h)

        img_size = w * sqr_size, h * sqr_size
        img = Image.new("RGB", size=img_size)
        draw = ImageDraw.Draw(img)

        for y in range(0, h * sqr_size, sqr_size):
            for x in range(0, w * sqr_size, sqr_size):
                draw.rectangle((x, y, x + sqr_size, y + sqr_size), outline=(31, 31, 31))

        x, y = self.BaseSnake.food
        x, y = int(x) * sqr_size, int(y) * sqr_size
        draw.rectangle(
            (x, y, x + sqr_size, y + sqr_size),
            fill=self.BaseSnake.COLORS["food"],
            outline=(31, 31, 31),
        )

        for i, (x, y) in enumerate(self.BaseSnake.snake):
            x, y = int(x) * sqr_size, int(y) * sqr_size
            if i == 0:
                draw.rectangle(
                    (x, y, x + sqr_size, y + sqr_size),
                    fill=self.BaseSnake.COLORS["head"],
                    outline=(31, 31, 31),
                )
            elif i == len(self.BaseSnake.snake) - 1:
                draw.rectangle(
                    (x, y, x + sqr_size, y + sqr_size),
                    fill=self.BaseSnake.COLORS["tail"],
                    outline=(31, 31, 31),
                )
            else:
                draw.rectangle(
                    (x, y, x + sqr_size, y + sqr_size),
                    fill=self.BaseSnake.COLORS["body"],
                    outline=(31, 31, 31),
                )

        if self.BaseSnake.limit_collision:
            draw.rectangle((0, 0, w, h), outline=(255, 255, 255), width=2)

        return img

    def render(self, mode="pygame", ):
        if self.render_mode in ["pygame", "human"]:
            self.BaseSnake.render()
        elif self.render_mode == "ascii":
            return self.generate_ascii()
            # time.sleep(0.5)
            # print("\033[H\033[J", end="")  # clear
        elif self.render_mode == "rgb_array":
            return self.generate_image()
        else:
            raise NotImplementedError(f"mode: {self.render_mode}")


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = SnakeEnv([5, 5])
    check_env(env, warn=True)
    print(f"env is apparently correct")
