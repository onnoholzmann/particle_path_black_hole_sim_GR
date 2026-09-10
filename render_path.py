import pygame
from calc_path import *
import time
import numpy as np

# setup
pygame.init()
w, h = 1280, 720
screen = pygame.display.set_mode((w, h))
clock = pygame.time.Clock()
running = True
font = pygame.font.SysFont("consolas", 20)
fps_history = []

# sun = Attractor(1e38, (w/2, h/2), 0.5)
# earth = Newtonian_object(sun, (w/2, h/4), (1.25, 0))
sun2 = Attractor(1, (w/2, h/2), 0.5, convert_a=True)
earth2 = Relative_object(sun2, (200, 0, np.pi/2), (0, 1.25, 0))

while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  # fill the screen with a color to wipe away anything from last frame
  screen.fill((50, 50, 50))

  # render
  # pygame.draw.circle(screen, (200, 200, 150), sun.pos, 20)
  # pygame.draw.circle(screen, (100, 200, 100), earth.update(5), 5)
  pygame.draw.circle(screen, (200, 200, 150), sun2.pos, 20)
  pygame.draw.circle(screen, (100, 200, 100), earth2.update(5), 5)

   # --- FPS Tracking ---
  clock.tick(60)  # limits FPS to 60
  current_fps = clock.get_fps()
  current_time = time.time()
  
  # Store current FPS with timestamp
  fps_history.append((current_time, current_fps))
  
  # Prune entries older than 10 seconds
  cutoff_time = current_time - 10.0
  fps_history = [item for item in fps_history if item[0] >= cutoff_time]
  
  # Calculate lowest FPS (ignore 0.0 which happens on the very first frame)
  valid_fps = [item[1] for item in fps_history if item[1] > 0]
  lowest_fps = min(valid_fps) if valid_fps else 0
  # Render FPS text
  fps_text = font.render(f"FPS: {current_fps:6.1f} | Lowest (10s): {lowest_fps:6.1f}", True, (255, 255, 255))
  screen.blit(fps_text, (10, 10))

  # flip() the display to put your work on screen
  pygame.display.flip()

  # clock.tick(60)  # limits FPS to 60

pygame.quit()