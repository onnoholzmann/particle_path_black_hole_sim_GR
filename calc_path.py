import scipy, math
from numba import njit
import numpy as np

"""
UNITS
mass is in KG
pos is in light seconds
speed is light seconds / second
time is in seconds

G needs to get converted into the right units
"""


# the attractor, like the sun or black hole
class Attractor:
  def __init__(self, mass, pos):
    # the mass can also be noted like 5e10
    self.mass = mass
    self.pos = np.array(pos)

G_light = scipy.constants.G / scipy.constants.c**3

@njit(fastmath=True)
def step_newt(speed, pos, attractor_pos, mass, time_passed, dt=1):
  current_speed = speed.copy()
  current_pos = pos.copy()

  for _ in range(time_passed):
    r = np.sqrt(np.sum((current_pos-attractor_pos)**2))
    # direction_vector = (attractor_pos - current_pos)/r
    # acceleration = G_light * mass / r**2
    # combined the /r and /r**2 for performance reasons and since they are multiplied in the current speed calc, it gives the same result
    direction_vector = (attractor_pos - current_pos)/r**3
    acceleration = G_light * mass
    current_speed += acceleration * dt * direction_vector
    current_pos += current_speed * dt

  return current_speed, current_pos

# the object, orbiting the attractor
class Newtonian_object:
  def __init__(self, attractor, pos, speed):
    self.attractor = attractor
    self.pos = np.array(pos, dtype=float)
    self.speed = np.array(speed, dtype=float)

  def update(self, time_passed):
    self.speed, self.pos = step_newt(self.speed, self.pos, self.attractor.pos, self.attractor.mass, time_passed)
    """
    for _ in range(time_passed):
      dt = 1
      r = np.sqrt(np.sum((self.pos-self.attractor.pos)**2))
      direction_vector = (self.attractor.pos - self.pos)/r

      acceleration = scipy.constants.G / scipy.constants.c**3 * self.attractor.mass / r**2
      self.speed += acceleration * dt * direction_vector
      self.pos += self.speed * dt

      # print(self.pos)
    """
    return self.pos