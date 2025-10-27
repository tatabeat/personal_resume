import pygame
import sys
from pygame.locals import *
import subprocess

def read_and_sort_rank(file_path):
    # 讀取文件並將數字存入list
    with open(file_path, 'r') as file:
        ranks = [int(line.strip()) for line in file]

    # 將list從大到小排序
    ranks.sort(reverse=True)

    # 將排序好的list存回文件
    with open(file_path, 'w') as file:
        for rank in ranks:
            file.write(f"{rank}\n")

    # 回傳前5個數字
    return ranks[:5]

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


# 讀取並排序rank.txt
top_5_ranks = read_and_sort_rank('rank.txt')

# 主循環
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_b:
                running = False
                pygame.quit()  # 關閉視窗
                subprocess.run(["python", "cover.py"])
                sys.exit()
    # 填充背景顏色
    screen.fill(BLACK)
    background = pygame.image.load('img/background1.jpg')
    screen.fill(BLACK)
    screen.blit(background, (-550, 0))
    font = pygame.font.Font(None, 96)  # 使用默認字體
    word = font.render("~RANK~", True, (25,200,200))
    screen.blit(word, (100, 75))
    font = pygame.font.Font(None, 72)
    pygame.draw.rect(screen, WHITE, (50, 150, 350, 450))

    # 在矩形中顯示前5名數字
    y_offset = 180  # 初始y座標
    for i, rank in enumerate(top_5_ranks, start=1):
        rank_text = font.render(f"{i}. {rank}", True, BLACK)
        screen.blit(rank_text, (100, y_offset))
        y_offset += 80  # 調整y座標以顯示下一個數字

    pygame.draw.rect(screen, WHITE, (100, 650, 250, 100))
    word = font.render("BACK [B]", True, BLACK)
    screen.blit(word, (115, 675))

    # 更新螢幕
    pygame.display.flip()

# 結束 Pygame
pygame.quit()
sys.exit()
