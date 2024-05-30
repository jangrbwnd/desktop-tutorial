import pygame
import math
import random
import time

# 화면 크기 상수
SIZE_OF_THE_SCREEN = 600, 645
# 벽돌 크기
HEIGHT_OF_BRICK = 18
WIDTH_OF_BRICK = 48
# 패들 크기
HEIGH_OF_PADDLE = 12
PADDLE_WIDTH = 100
# 패들의 Y 좌표
PADDLE_Y = SIZE_OF_THE_SCREEN[1] - HEIGH_OF_PADDLE - 10
# 공의 크기
BALL_DIAMETER = 18
BALL_RADIUS = BALL_DIAMETER // 2
# 패들의 최대 X 좌표
MAX_PADDLE_X = SIZE_OF_THE_SCREEN[0] - PADDLE_WIDTH
# 공의 최대 X, Y 좌표
MAX_BALL_X = SIZE_OF_THE_SCREEN[0] - BALL_DIAMETER
MAX_BALL_Y = SIZE_OF_THE_SCREEN[1] - BALL_DIAMETER

# 색상 상수
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
COLOR_OF_BRICK = (153, 76, 0)
PADDLE_COLOR = (204, 0, 0)

FPS = 60
FPSCLOCK = pygame.time.Clock()

pygame.init()  # pygame 모듈 초기화
screen = pygame.display.set_mode(SIZE_OF_THE_SCREEN)
pygame.display.set_caption("BREAKOUT")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

class Breakout:

    def __init__(self):
        self.capture = 0
        self.radias = 40
        self.direction = random.randint(-45, 45)

        if 90 - self.radias <= (360 + self.direction)%360 < 90:
             self.direction = 70
        elif 90 <= (360 + self.direction)%360 < 90 + self.radias:
             self.direction = 90 + self.radias
        elif 270 - self.radias <=(360 + self.direction)%360 < 270:
             self.direction = 270-self.radias
        elif 270 <= (360 + self.direction)%360 < 270 + self.radias:
            self.direction = 270+ self.radias
        self.paddle = pygame.Rect(215, PADDLE_Y, PADDLE_WIDTH, HEIGH_OF_PADDLE)
        self.ball = pygame.Rect(225, PADDLE_Y - BALL_DIAMETER, BALL_DIAMETER, BALL_DIAMETER)
        self.balled_left = self.ball.left
        self.balled_top = self.ball.top
        self.speed = 15
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
            self.paddle.left -= 12
            if self.paddle.left < 0:
                self.paddle.left = 0
        if keys[pygame.K_RIGHT]:
            self.paddle.left += 12
            if self.paddle.left > MAX_PADDLE_X:
                self.paddle.left = MAX_PADDLE_X

    def ball_bounce(self, diff):
        self.direction = (180 - self.direction) % 360
        self.direction -= diff

        if 90 - self.radias <= (360 + self.direction) % 360 < 90:
            self.direction = 70
        elif 90 <= (360 + self.direction) % 360 < 90 + self.radias:
            self.direction = 90 + self.radias
        elif 270 - self.radias <= (360 + self.direction) % 360 < 270:
            self.direction = 270 - self.radias
        elif 270 <= (360 + self.direction) % 360 < 270 + self.radias:
            self.direction = 270 + self.radias

    def move_ball(self):
        self.direction_radians = math.radians(self.direction)
        self.ball.left += self.speed * math.sin(self.direction_radians)
        self.ball.top -= self.speed * math.cos(self.direction_radians)

        if self.ball.left <= 0:
            self.balled_left = self.ball.left
            self.balled_top = self.ball.top
            self.direction = (360 - self.direction) % 360
            if 90 - self.radias <= (360 + self.direction) % 360 < 90:
                self.direction = 70
            elif 90 <= (360 + self.direction) % 360 < 90 + self.radias:
                self.direction = 90 + self.radias
            elif 270 - self.radias <= (360 + self.direction) % 360 < 270:
                self.direction = 270 - self.radias
            elif 270 <= (360 + self.direction) % 360 < 270 + self.radias:
                self.direction = 270 + self.radias
            self.ball.left = 0

        elif self.ball.left >= MAX_BALL_X:
            self.balled_left = self.ball.left
            self.balled_top = self.ball.top
            self.direction = (360 - self.direction) % 360
            if 90 - self.radias <= (360 + self.direction) % 360 < 90:
                self.direction = 70
            elif 90 <= (360 + self.direction) % 360 < 90 + self.radias:
                self.direction = 90 + self.radias
            elif 270 - self.radias <= (360 + self.direction) % 360 < 270:
                self.direction = 270 - self.radias
            elif 270 <= (360 + self.direction) % 360 < 270 + self.radias:
                self.direction = 270 + self.radias
            self.ball.left = MAX_BALL_X

        if self.ball.top < 0:
            self.balled_left = self.ball.left
            self.balled_top = self.ball.top
            self.ball_bounce(0)
            self.ball.top = 0

        elif self.ball.top >= MAX_BALL_Y:
            self.balled_left = self.ball.left
            self.balled_top = self.ball.top
            self.ball_bounce(0)
            self.ball.top = MAX_BALL_Y

    def caculate(self, x, x1, x2, y1, y2):
        if x2 - x1 == 0:
            return float('inf')
        slope = (y2 - y1) / (x2 - x1) + 0.001
        y_intercept = y1 - slope * x1
        return slope * x + y_intercept

    def take_action(self):
        pygame.event.pump()
        reward = 0.0001
        terminal = False

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
                    if self.caculate(brick.x, self.balled_left + BALL_DIAMETER, self.ball.left + BALL_DIAMETER, self.balled_top, self.ball.top) >= brick.y - (HEIGHT_OF_BRICK):
                        self.balled_left = self.ball.left
                        self.balled_top = self.ball.top
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break
                    elif brick.y < self.caculate(brick.x, self.balled_left + BALL_DIAMETER, self.ball.left + BALL_DIAMETER, self.balled_top, self.ball.top) < brick.y + (HEIGHT_OF_BRICK):
                        self.balled_left = self.ball.left
                        self.balled_top = self.ball.top
                        self.direction = (360 - self.direction) % 360
                        if 90 - self.radias <= (360 + self.direction) % 360 < 90:
                            self.direction = 70
                        elif 90 <= (360 + self.direction) % 360 < 90 + self.radias:
                            self.direction = 90 + self.radias
                        elif 270 - self.radias <= (360 + self.direction) % 360 < 270:
                            self.direction = 270 - self.radias
                        elif 270 <= (360 + self.direction) % 360 < 270+ self.radias:
                            self.direction = 270 + self.radias

                        self.bricks.remove(brick)
                        break
                    else:
                        self.balled_left = self.ball.left
                        self.balled_top = self.ball.top
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break
                else:
                    if self.caculate(brick.x + WIDTH_OF_BRICK, self.balled_left, self.ball.left, self.balled_top, self.ball.top) >= brick.y + (HEIGHT_OF_BRICK):
                        self.balled_left = self.ball.left
                        self.balled_top = self.ball.top
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break
                    elif brick.y < self.caculate(brick.x + WIDTH_OF_BRICK, self.balled_left, self.ball.left, self.balled_top, self.ball.top) < brick.y + (HEIGHT_OF_BRICK):
                        self.balled_left = self.ball.left
                        self.balled_top = self.ball.top
                        self.direction = (360 - self.direction) % 360
                        if 90 - self.radias <= (360 + self.direction) % 360 < 90:
                            self.direction = 70
                        elif 90 <= (360 + self.direction) % 360 < 90 + self.radias:
                            self.direction = 90 + self.radias
                        elif 270 - self.radias <= (360 + self.direction) % 360 < 270:
                            self.direction = 270 - self.radias
                        elif 270 <= (360 + self.direction) % 360 < 270 + self.radias:
                            self.direction = 270 + self.radias
                        self.bricks.remove(brick)
                        break
                    else:
                        self.balled_left = self.ball.left
                        self.balled_top = self.ball.top
                        self.ball_bounce(0)
                        self.bricks.remove(brick)
                        break

        if len(self.bricks) == 0:
            self.terminal = True
            self.__init__()
        if self.ball.colliderect(self.paddle):
            reward = 2.0
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            paddle = self.paddle.left
            ball = self.ball.left
            diff = (paddle + PADDLE_WIDTH / 2) - (ball + BALL_DIAMETER / 2)
            self.ball_bounce(diff)
        elif self.ball.top > self.paddle.top:
            terminal = True
            self.__init__()
            reward = -1

        self.draw_bricks()
        self.draw_ball()
        self.draw_paddle()
        elapsed_time = int(120 - (time.time() - self.start_time))
        if elapsed_time <= 0:
            self.terminal = True
            elapsed_time = 0
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        time_text = font.render(f'Time: {elapsed_time}s', True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(time_text, (SIZE_OF_THE_SCREEN[0] - 150, 10))

        # 화면 캡처 및 image_data에 할당
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # 화면 업데이트
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal

# 게임 실행
if __name__ == "__main__":
    game = Breakout()
    screen.fill(BLACK)
    
    for i in range(3, 0, -1):
        screen.fill(BLACK)
        countdown_text = font.render(f'{i}', True, WHITE)
        screen.blit(countdown_text, (SIZE_OF_THE_SCREEN[0] // 2, SIZE_OF_THE_SCREEN[1] // 2))
        pygame.display.update()
        time.sleep(1)
    
    running = True
    score_list=[]
    while running:
        image_data, reward, terminal = game.take_action()
        score = game.score
        if len(score_list) < 2:
            score_list.append(score)
        elif len(score_list) ==2:
            score_list[0] = score_list[1]
            score_list[1] = score
        if terminal:
            # "Finish" 메시지 표시
            game_over_text = font.render('Finish', True, WHITE)
            screen.blit(game_over_text, (SIZE_OF_THE_SCREEN[0] // 2 - 60, SIZE_OF_THE_SCREEN[1] // 2))
            pygame.display.update()
            
            # 사용자가 창을 닫을 때까지 대기
            while True:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        quit()
                # 점수와 남은 시간 계속 표시
                score_text = font.render(f'Score: {score_list[0]}', True, WHITE)
                #elapsed_time = int(120 - (time.time() - game.start_time))
                #if elapsed_time <= 0:
                 #   elapsed_time = 0
                #time_text = font.render(f'Time: {elapsed_time}s', True, WHITE)
                screen.blit(score_text, (SIZE_OF_THE_SCREEN[0] // 2 - 60, SIZE_OF_THE_SCREEN[1] // 2 + 30))
                #screen.blit(time_text, (SIZE_OF_THE_SCREEN[0] - 150, 10))
                pygame.display.update()
