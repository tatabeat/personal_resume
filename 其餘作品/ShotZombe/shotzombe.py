import pygame
import random
import sys
from pygame.locals import *
import subprocess
import time

def save_score_to_rank(file_path, score):
    with open(file_path, 'a') as file:
        file.write(f"{score}\n")

# 初始化 Pygame
pygame.init()
pygame.mixer.init()  # 初始化混音器

# 設定螢幕大小和標題
size = (450, 800)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("ShotZombie Game")

# 設定顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORIGINAL_COLOR = (255, 255, 255)  # 原色
RED = (255, 0, 0)  # 紅色
LIGHT_PURPLE = (218, 112, 214)  # 淺紫色
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)  # 黃色

# 定義一些變數
positions = [(75, 740), (225, 740), (375, 740)]  # 判定點位置
zombies = []  # 儲存僵屍的清單
score = 0  # 分數
time_limit = 34  # 時間限制為34秒
font = pygame.font.Font(None, 72)  # 倒數計時字型
start_ticks = pygame.time.get_ticks()  # 記錄開始時間
penalty_end_time = 0  # 懲罰時間結束
zombie_count = 0  # 計算產生的僵屍數量

# 顏色列表，用於循環
colors = [ORIGINAL_COLOR, RED, LIGHT_PURPLE, GREEN, YELLOW]

# 載入背景圖片
background_image = pygame.image.load('img/background.png')

# 載入音樂
pygame.mixer.music.load('bgm/bgm.mp3')
pygame.mixer.music.play(-1)  # 無限循環播放背景音樂

# 載入音效
gum_sound = pygame.mixer.Sound('bgm/gum.mp3')
jump_sound = pygame.mixer.Sound('bgm/jump.mp3')  # 新增的跳躍音效
start_sound = pygame.mixer.Sound('bgm/start.mp3')  # 新增的開始音效
end_sound = pygame.mixer.Sound('bgm/end.wav')  # 新增的結束音效

# 載入僵屍圖片
zombie_image = pygame.image.load('img/image.png')

def tint_image(image, color):
    """給圖片添加顏色"""
    tinted_image = image.copy()
    tinted_image.fill(color, special_flags=pygame.BLEND_MULT)
    return tinted_image

# 定義僵屍類別
class Zombie(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = tint_image(zombie_image, color)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.jump = False

    def move_down(self):
        self.rect.y += 150

# 初始化僵屍群
for i in range(5):
    row = i % 3
    x, y = positions[row]
    color = colors[zombie_count // 50 % len(colors)]  # 根據僵屍數量來選擇顏色
    zombie = Zombie(x, y - 600 + i * 150, color)
    zombies.append(zombie)
    zombie_count += 1

# 主迴圈
running = True
game_started = False  # 用來判斷遊戲是否已經開始
while running:
    screen.blit(background_image, (0, 0))  # 繪製背景圖片

    # 檢查事件
    current_time = pygame.time.get_ticks()
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN and game_started and current_time > penalty_end_time:
            if event.key == K_v:
                hit_pos = 0
            elif event.key == K_b:
                hit_pos = 1
            elif event.key == K_n:
                hit_pos = 2
            else:
                hit_pos = -1

            if hit_pos != -1:
                hit_success = False
                closest_zombie = None
                min_distance = float('inf')
                for zombie in zombies:
                    if positions[hit_pos][1] - 60 <= zombie.rect.centery <= positions[hit_pos][1]:
                        if zombie.rect.centerx == positions[hit_pos][0]:
                            zombies.remove(zombie)
                            for z in zombies:
                                z.move_down()
                            random_row = random.randint(0, 2)
                            x, y = positions[random_row]
                            color = colors[zombie_count // 50 % len(colors)]  # 根據僵屍數量來選擇顏色
                            new_zombie = Zombie(x, y - 600, color)
                            zombies.append(new_zombie)
                            zombie_count += 1
                            gum_sound.play()  # 播放音效
                            score += 50
                            hit_success = True
                            break

                    distance = abs(zombie.rect.centery - positions[hit_pos][1])
                    if distance < min_distance:
                        min_distance = distance
                        closest_zombie = zombie

                if not hit_success and closest_zombie:
                    closest_zombie.jump = True
                    jump_sound.play()  # 播放跳躍音效
                    penalty_end_time = current_time + 500  # 0.5秒懲罰時間

    if not game_started:
        countdown_time = 3 - (pygame.time.get_ticks() - start_ticks) // 1000
        if countdown_time > 0:
            countdown_text = font.render(str(countdown_time), True, WHITE)
            screen.blit(countdown_text, (size[0] // 2 - countdown_text.get_width() // 2, size[1] // 2 - countdown_text.get_height() // 2))
        else:
            start_text = font.render("Start", True, WHITE)
            screen.blit(start_text, (size[0] // 2 - start_text.get_width() // 2, size[1] // 2 - start_text.get_height() // 2))
            start_sound.play()  # 播放開始音效
            pygame.display.flip()
            pygame.time.wait(1000)
            game_started = True
            start_ticks = pygame.time.get_ticks()  # 重新記錄開始時間
    else:
        # 繪製僵屍
        for zombie in zombies:
            if zombie.jump and current_time < penalty_end_time:
                if (current_time // 100) % 2 == 0:
                    screen.blit(zombie.image, (zombie.rect.x, zombie.rect.y - 20))  # 模擬跳躍效果
                else:
                    screen.blit(zombie.image, zombie.rect)
            else:
                screen.blit(zombie.image, zombie.rect)
                zombie.jump = False

        # 計算並顯示時間和分數
        seconds = (pygame.time.get_ticks() - start_ticks) / 1000
        if seconds > time_limit:
            running = False
        else:
            time_text = pygame.font.Font(None, 36).render(f"Time: {int(time_limit - seconds)}", True, WHITE)
            screen.blit(time_text, (10, 10))
            score_text = pygame.font.Font(None, 36).render(f"Score: {score}", True, WHITE)
            screen.blit(score_text, (10, 50))

    pygame.display.flip()  # 更新畫面

# 遊戲結束，停止背景音樂
pygame.mixer.music.stop()

# 播放結束音效
end_sound.play()

# 顯示最終分數
screen.fill(BLACK)
final_score_text = pygame.font.Font(None, 36).render(f"Final Score: {score}", True, WHITE)
screen.blit(final_score_text, (100, 400))
pygame.display.flip()
save_score_to_rank('rank.txt', score)
pygame.time.wait(3000)
pygame.quit()
subprocess.run(["python", "cover.py"])
sys.exit()
