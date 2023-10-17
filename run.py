import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
import numpy
import matplotlib.pyplot as plt
import sys
import os

from model import get_result
import pygame

img_size = 28
image = [[0.0] * img_size for i in range(img_size)]
pygame.init()
screen_size = 800
clear_button_width = 300
clear_button_height = 100
number_guess_period = 1000
clear_button_pos = (screen_size, 0)
font = pygame.font.Font(None, 32)
bold_delta = 0.2
block_size = screen_size // img_size
screen = pygame.display.set_mode([screen_size + clear_button_width, screen_size])
running = True
screen.fill((255, 255, 255))
pygame.draw.rect(screen, (255, 0, 0),
                 pygame.Rect(clear_button_pos[0], clear_button_pos[1], clear_button_width, clear_button_height))
clear_text = font.render("Clear", True, (0, 0, 0))
clear_text_rect = clear_text.get_rect()
screen.blit(clear_text, (clear_button_pos[0] + (clear_button_width - clear_text_rect.width) // 2,
                         clear_button_pos[1] + (clear_button_height - clear_text_rect.height) // 2))


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 8 * 8, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 8 * 8)
        output = self.fc1(output)

        return output


model = Network()
model.load_state_dict(torch.load(resource_path('model/v4.pth'), map_location=torch.device('cpu')))
model.eval()

def get_result(image):
    np_array = np.array(image).transpose()
    img_to_model = np.reshape(np_array, (1, 1, 28, 28))
    outputs = model(torch.Tensor(img_to_model))
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


def draw_point_sides(x, y):
    pygame.draw.rect(screen, (230, 230, 230), pygame.Rect(x * block_size, y * block_size, block_size, block_size),
                     1)


for x in range(img_size):
    for y in range(img_size):
        draw_point_sides(x, y)


def draw_point(x, y):
    pygame.draw.rect(screen,
                     (int(255 * (1 - image[x][y])), int(255 * (1 - image[x][y])), int(255 * (1 - image[x][y]))),
                     pygame.Rect(x * block_size, y * block_size, block_size, block_size))


def put_point(x, y, add_delta=bold_delta):
    if x < 0 or y < 0 or add_delta <= 0 or x >= img_size or y >= img_size:
        return
    image[x][y] = min(1.0, image[x][y] + add_delta)
    draw_point(x, y)
    put_point(x, y - 1, add_delta - bold_delta * 4 / 5)
    put_point(x, y + 1, add_delta - bold_delta * 4 / 5)
    put_point(x - 1, y, add_delta - bold_delta * 4 / 5)
    put_point(x + 1, y, add_delta - bold_delta * 4 / 5)
    put_point(x - 1, y - 1, add_delta - bold_delta * 6 / 7)
    put_point(x + 1, y + 1, add_delta - bold_delta * 6 / 7)
    put_point(x - 1, y + 1, add_delta - bold_delta * 6 / 7)
    put_point(x + 1, y - 1, add_delta - bold_delta * 6 / 7)


def draw_answer():
    answer = str(get_result(image))
    text = font.render(answer, True, (0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(screen_size, clear_button_height, 32, 32))
    screen.blit(text, (screen_size, clear_button_height))


def clear_image():
    for i in range(img_size):
        for j in range(img_size):
            image[i][j] = 0
            draw_point(i, j)
            draw_point_sides(i, j)


iteration = 0
while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            if clear_button_pos[0] <= mouse_pos[0] <= clear_button_pos[0] + clear_button_width and clear_button_pos[
                1] <= mouse_pos[1] <= clear_button_pos[1] + clear_button_height:
                clear_image()
    if pygame.mouse.get_pressed()[0]:
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos[0] <= screen_size and mouse_pos[1] <= screen_size:
            x = min(img_size - 1, img_size - 1, mouse_pos[0] // block_size)
            y = min(img_size - 1, img_size - 1, mouse_pos[1] // block_size)
            put_point(x, y)
    # Flip the display
    iteration += 1
    if iteration % number_guess_period == 0:
        draw_answer()
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
