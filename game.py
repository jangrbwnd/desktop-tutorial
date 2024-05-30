import pygame
import math
import random
import time

# Constant for determining the screen size. 
SIZE_OF_THE_SCREEN = 600, 645
# Dimensions of the bricks
HEIGHT_OF_BRICK  = 18
WIDTH_OF_BRICK   = 48
# Dimensions of the paddle
HEIGHT_OF_PADDLE = 12
PADDLE_WIDTH  = 75
# Y coordinate for Paddle
PADDLE_Y = SIZE_OF_THE_SCREEN[1] - HEIGHT_OF_PADDLE - 10
# Dimensions of the ball
BALL_DIAMETER = 18
BALL_RADIUS   = BALL_DIAMETER // 2
# X coordinate for Paddles
MAX_PADDLE_X = SIZE_OF_THE_SCREEN[0] - PADDLE_WIDTH
MAX_BALL_X   = SIZE_OF_THE_SCREEN[0] - BALL_DIAMETER
MAX_BALL_Y   = SIZE_OF_THE_SCREEN[1] - BALL_DIAMETER

# Color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE  = (0, 0, 255)
COLOR_OF_BRICK = (153, 76, 0)
PADDLE_COLOR = (204,0,0)

FPS = 60
FPSCLOCK = pygame.time.Clock()

pygame.init() # Calling pygame module
screen = pygame.display.set_mode(SIZE_OF_THE_SCREEN)
pygame.display.set_caption("BREAKOUT")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

class Breakout:

    def __init__(self):
        self.capture = 0
        self.direction = random.randint(-45, 45)
        if self.direction == 0 or self.direction == 90 or self.direction == 180 or self.direction == 270:
            self.direction += 4
        self.paddle = pygame.Rect(215, PADDLE_Y, PADDLE_WIDTH, HEIGHT_OF_PADDLE)
        self.ball = pygame.Rect(225, PADDLE_Y - BALL_DIAMETER, BALL_DIAMETER, BALL_DIAMETER)
        self.balled_left = self.ball.left
        self.balled_top = self.ball.top
        self.speed = 20
        self.create_bricks()
        self.start_time = time.time()
        self.score = 0
        self.terminal = False

    def create_bricks(self):
        y_ofs = 50
        self.bricks = []
        for i in range(11):
            x_ofs = 10
            for j in range(12):
                self.bricks.append(pygame.Rect(x_ofs, y_ofs, WIDTH_OF_BRICK, HEIGHT_OF_BRICK))
                x_ofs += WIDTH_OF_BRICK 
            y_ofs += HEIGHT_OF_BRICK 

    def draw_bricks(self):
        for brick in self.bricks:
            pygame.draw.rect(screen, COLOR_OF_BRICK, brick)

    def draw_paddle(self):
        pygame.draw.rect(screen, PADDLE_COLOR, self.paddle)

    def draw_ball(self):
        pygame.draw.circle(screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)
        
    def check_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.paddle.left -= 24
            if self.paddle.left < 0:
                self.paddle.left = 0
        if keys[pygame.K_RIGHT]:
            self.paddle.left += 24
            if self.paddle.left > MAX_PADDLE_X:
                self.paddle.left = MAX_PADDLE_X

    def ball_bounce(self, diff):
        self.direction = (180 - self.direction) % 360
        self.direction -= diff
        if self.direction == 0 or self.direction == 90 or self.direction == 180 or self.direction == 270:
            self.direction += 4

    def move_ball(self):
        self.direction_radians = math.radians(self.direction)
        self.ball.left += self.speed * math.sin(self.direction_radians)
        self.ball.top -= self.speed * math.cos(self.direction_radians)

        if self.ball.left <= 0:
            self.direction = (360 - self.direction) % 360
            if self.direction == 0 or self.direction == 90 or self.direction == 180 or self.direction == 270:
                self.direction += 4
            self.ball.left = 0
        elif self.ball.left >= MAX_BALL_X:
            self.direction = (360 - self.direction) % 360
            if self.direction == 0 or self.direction == 90 or self.direction == 180 or self.direction == 270:
                self.direction += 4
            self.ball.left = MAX_BALL_X
        if self.ball.top < 0:
            self.ball_bounce(0)
            self.ball.top = 0
        elif self.ball.top >= MAX_BALL_Y:
            self.ball_bounce(0)
            self.ball.top = MAX_BALL_Y

    def calculate(self, x, x1, x2, y1, y2):
        if x2 - x1 == 0:
            return float('inf')  # 수직선을 나타내기 위해 큰 값을 반환
        slope = (y2 - y1) / (x2 - x1) + 0.001
        y_intercept = y1 - slope * x1
        return slope * x + y_intercept

    def take_action(self):
        pygame.event.pump()
        reward = 0.01

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill(BLACK)
        self.check_input()
        self.move_ball()

        for brick in self.bricks:
            if self.ball.colliderect(brick):
                reward = 1
                self.score += 1
                x_dif = self.ball.left - self.balled_left
                y_dif = self.ball.top + self.balled_top
                if (x_dif and y_dif >= 0) or (x_dif >= 0 and y_dif < 0):
                    if self.calculate(brick.x, self.balled_left + BALL_DIAMETER, self.ball.left + BALL_DIAMETER, self.balled_top, self.ball.top) >= brick.y - HEIGHT_OF_BRICK:
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break
                    elif brick.y < self.calculate(brick.x, self.balled_left + BALL_DIAMETER, self.ball.left + BALL_DIAMETER, self.balled_top, self.ball.top) < brick.y + HEIGHT_OF_BRICK:
                        self.direction = (360 - self.direction) % 360
                        if self.direction == 0 or self.direction == 90 or self.direction == 180 or self.direction == 270:
                            self.direction += 4
                        self.bricks.remove(brick)
                        break
                    else:
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break
                else:
                    if self.calculate(brick.x + WIDTH_OF_BRICK, self.balled_left, self.ball.left, self.balled_top, self.ball.top) >= brick.y + HEIGHT_OF_BRICK:
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break
                    elif brick.y < self.calculate(brick.x + WIDTH_OF_BRICK, self.balled_left, self.ball.left, self.balled_top, self.ball.top) < brick.y + HEIGHT_OF_BRICK:
                        self.direction = (360 - self.direction) % 360
                        if self.direction == 0 or self.direction == 90 or self.direction == 180 or self.direction == 270:
                            self.direction += 4
                        self.bricks.remove(brick)
                        break
                    else:
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break

        if len(self.bricks) == 0:
            self.terminal = True
            self.__init__()
        if self.ball.colliderect(self.paddle):
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            paddle = self.paddle.left
            ball = self.ball.left
            diff = (paddle + PADDLE_WIDTH / 2) - (ball + BALL_DIAMETER / 2)
            self.ball_bounce(diff)
        elif self.ball.top > self.paddle.top:
            self.terminal = True
            self.__init__()
            reward = -1

        self.draw_bricks()
        self.draw_ball()
        self.draw_paddle()
        
        elapsed_time = int(120 - (time.time() - self.start_time))
        if elapsed_time <= 0:
            self.terminal = True
            elapsed_time = 0
        
        # Draw score and time
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        time_text = font.render(f'Time: {elapsed_time}s', True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(time_text, (SIZE_OF_THE_SCREEN[0] - 150, 10))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

        return reward, self.terminal

def main():
    pygame.init()
    screen = pygame.display.set_mode(SIZE_OF_THE_SCREEN)
    pygame.display.set_caption("BREAKOUT")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)

    game = Breakout()
    
    running = True
    while running:
        reward, terminal = game.take_action()
        if terminal:
            running = False

    pygame.quit()

main()
