import pygame
import sys
from pygame.locals import *
import subprocess

# 初始化 Pygame
pygame.init()
# 設定螢幕大小和標題
size = (450, 800)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("ShotZombie Game")

# 設定顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY=(150,150,150)

# 主循環
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_v:
                running = False
                pygame.quit()  #關閉視窗
                subprocess.run(["python", "shotzombe.py"])
                sys.exit()
            elif event.key == K_n:
                running = False
                pygame.quit()  #關閉視窗
                subprocess.run(["python", "rank.py"])
                sys.exit()
    # 填充背景顏色
    background=pygame.image.load('img/background1.jpg')
    screen.fill(BLACK)
    screen.blit(background,(0,0))
    font=pygame.font.Font(None,72)  # 使用默認字體
    word=font.render("\"ShotZombe\"", True,(25,200,200))
    screen.blit(word,(75,150))
    font=pygame.font.Font(None,72)  
    pygame.draw.rect(screen, WHITE, (100, 400, 250, 100))
    word=font.render("START [V]", True, BLACK)
    screen.blit(word,(110,425))
    pygame.draw.rect(screen, WHITE, (100, 550, 250, 100))
    word=font.render("RANK [N]", True, BLACK)
    screen.blit(word,(110,575))
    
    # 更新螢幕
    pygame.display.flip()

# 結束 Pygame
pygame.quit()
sys.exit()
