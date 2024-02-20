import numpy as np
import pygame
from pygame.math import Vector2

# Base snake game


class SnakeGame:

    COLORS = {
        "head": (0, 255, 0),
        "body": (255, 255, 255),
        "tail": (255, 0, 0),
        "food": (0, 0, 255) 
    }
    DIRECTIONS = {
        pygame.K_LEFT: Vector2(-1, 0),
        pygame.K_UP: Vector2(0, -1),
        pygame.K_RIGHT: Vector2(1, 0),
        pygame.K_DOWN: Vector2(0, 1),
    }

    def __init__(self, shape, seed=None, limit_collision=False, fps=60, sqr_size=40):
        self._shape = Vector2(shape)
        self.WIDTH: int = shape[0]
        self.HEIGHT: int = shape[1]
        self.limit_collision = limit_collision
        self.fps = fps
        self.sqr_size = sqr_size
        self.hidden = True
        np.random.seed(seed)

        self.game_over = False
        self.speed = 2  # squares per second / for the loop
        self.snake_dir = Vector2(1, 0)
        self.new_direction = None
        self.dtime = 0
        self.last_pos = None
        self.snake = [Vector2(self.WIDTH, self.HEIGHT) // 2]
        self.snake.append(self.snake[0].copy() +
                          self.DIRECTIONS[pygame.K_LEFT])
        self.ate_last_frame = False
        self.new_food()

    def init_pygame(self):
        pygame.init()

        self.surface = pygame.display.set_mode(
            (self.WIDTH * self.sqr_size, self.HEIGHT * self.sqr_size), flags=pygame.HIDDEN)
        self.clock = pygame.time.Clock()

    def start(self, loop=True):

        self.init_pygame()
        if loop:
            self.loop()

    def loop(self):
        self.running = True
        while self.running:
            self.handle_input()
            self.update()
            self.render()

        self.close()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                if event.key == pygame.K_r:
                    self.reset()
                if event.key in self.DIRECTIONS.keys():
                    self.new_direction = self.DIRECTIONS.get(event.key)
                if event.key == pygame.K_PLUS:
                    self.speed += 0.5
                if event.key == pygame.K_MINUS:
                    self.speed -= 0.5
                if event.key == pygame.K_KP_6:
                    self.turn_dir("right")
                if event.key == pygame.K_KP_4:
                    self.turn_dir("left")

    def update(self, is_step=False):
        if self.game_over:
            return
        if not is_step:
            self.dtime += self.clock.get_time() / 1000
            if (self.dtime == 0 or 1/self.dtime > self.speed):
                return

        self.dtime = 0
        self.change_dir(self.new_direction)
        self.last_pos = self.snake[0]
        new_head = self.snake[0] + self.snake_dir

        if self.limit_collision:
            if self.get_collision(new_head) == "wall":
                self.running = False
                self.game_over = True
                if not is_step:
                    print("Game Over")
                return
        else:
            new_head = Vector2(
                [new_head.x % self.WIDTH, new_head.y % self.HEIGHT])

        if self.get_collision(new_head) == "food":
            self.snake.insert(0, new_head)
            self.ate_last_frame = True
            self.new_food()
        else:
            self.ate_last_frame = False
            self.snake.insert(0, new_head)
            self.snake.pop()

        if self.get_collision(new_head) == "snake":
            self.running = False
            self.game_over = True
            if not is_step:
                print("Game Over")
            return

    def get_collision(self, new_head: Vector2):
        if new_head == self.food:
            return "food"
        if new_head in self.snake[1:]:
            return "snake"
        if not self.limit_collision:
            return None
        if new_head[0] not in range(0, self.WIDTH) or new_head[1] not in range(0, self.HEIGHT):
            return "wall"

    def get_collision_distance(self, dir, coll_types=["snake", "wall"], max_steps=-1):
        temp_head = self.snake[0].copy()
        distance = 0
        max_steps = self.get_related_max(dir) if max_steps < 0 else max_steps
        n = 0
        while n < max_steps and self.get_collision(temp_head + dir) not in coll_types:
            n += 1
            distance += 1
            temp_head += dir
            temp_head = Vector2(temp_head.x % self.WIDTH, temp_head.y % self.HEIGHT)
        return distance

    def get_related_max(self, dir: Vector2):
        HORIZONTAL = [self.DIRECTIONS[pygame.K_RIGHT],
                               self.DIRECTIONS[pygame.K_LEFT]]
        VERTICAL = [self.DIRECTIONS[pygame.K_UP],
                             self.DIRECTIONS[pygame.K_DOWN]]
        if dir in HORIZONTAL:
            return self.WIDTH
        if dir in VERTICAL:
            return self.HEIGHT
        return 0

    def change_dir(self, dir: Vector2 | None):
        if dir is None:
            return
        dot = self.snake_dir.dot(dir)
        if dot != 0:
            return
        self.snake_dir = dir

    def turn_dir(self, dir):
        self.snake_dir = self._get_turn_dir(dir)

    def _get_turn_dir(self, dir):
        key = [key for key in self.DIRECTIONS if (
            np.array(self.DIRECTIONS[key]) == np.array(self.snake_dir)).all()]
        idx = list(self.DIRECTIONS).index(key[0])
        if dir == "left":
            new_dir = list(self.DIRECTIONS.values())[(idx-1) % 4]
            return new_dir
        if dir == "right":
            new_dir = list(self.DIRECTIONS.values())[(idx+1) % 4]
            return new_dir
        return self.snake_dir

    def new_food(self):
        shape = np.array([self.WIDTH, self.HEIGHT])
        new_food = (np.random.rand(2) * shape).astype(np.int32)
        new_food = Vector2(new_food.tolist())

        while new_food in self.snake:
            new_food = (np.random.rand(2) * shape).astype(np.int32)
            new_food = Vector2(new_food.tolist())
        self.food = new_food

    def render(self):

        if self.hidden:
            self.hidden = False
            self.surface = pygame.display.set_mode(
                self.surface.get_size(), flags=pygame.SHOWN)

        self.surface.fill(
            (0, 0, 0), (0, 0, self.surface.get_width(), self.surface.get_height()))
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                pygame.draw.rect(self.surface, (31, 31, 31),
                                 (i * self.sqr_size, j * self.sqr_size, self.sqr_size, self.sqr_size), 1)

        x, y = self.food * self.sqr_size
        x, y = int(x), int(y)
        self.surface.fill(self.COLORS["food"], (x, y, self.sqr_size, self.sqr_size))

        for i, (x, y) in enumerate(self.snake):
            x, y = x * self.sqr_size, y * self.sqr_size
            x, y = int(x), int(y)
            if i == 0:
                self.surface.fill(
                    self.COLORS["head"], (x, y, self.sqr_size, self.sqr_size))
            elif i == len(self.snake) - 1:
                self.surface.fill(
                    self.COLORS["tail"], (x, y, self.sqr_size, self.sqr_size))
            else:
                self.surface.fill(
                    self.COLORS["body"], (x, y, self.sqr_size, self.sqr_size))

        if self.limit_collision:
            pygame.draw.rect(self.surface, (255, 255, 255), (0, 0,
                             self.WIDTH*self.sqr_size, self.HEIGHT*self.sqr_size), 2)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def reset(self):
        self.game_over = False
        self.speed = 2  # squares per second / for the loop
        self.snake_dir = Vector2(1, 0)
        self.new_direction = None
        self.dtime = 0
        self.last_pos = None
        self.snake = [Vector2(self.WIDTH, self.HEIGHT) // 2]
        self.snake.append(self.snake[0].copy() +
                          self.DIRECTIONS[pygame.K_LEFT])
        self.ate_last_frame = False
        self.new_food()

    def close(self):
        pygame.display.quit()


if __name__ == "__main__":
    game = SnakeGame([10, 5], limit_collision=False)
    game.start()
