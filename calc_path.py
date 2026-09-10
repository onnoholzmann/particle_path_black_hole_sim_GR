import scipy, math
from numba import njit
import numpy as np

"""
UNITS
mass is in KG or 
pos is in light seconds
speed is light seconds / second
time is in seconds

G needs to get converted into the right units
"""


# the attractor, like the sun or black hole
class Attractor:
  def __init__(self, mass, pos, spin_parameter, convert_mass=False, convert_a=False):
    # the mass can also be noted like 5e10
    self.pos = np.array(pos)
    self.a = spin_parameter
    self.mass = mass
    if convert_mass:
      self.mass *= G_light
      # convert mass to geometric units in light seconds
    if convert_a:
      self.a *= self.mass
      # convert the unitless spin parameter into angular momentum
      

G_light = scipy.constants.G / scipy.constants.c**3

@njit(fastmath=True)
def step_newt(speed, pos, attractor_pos, mass, time_passed, dt=1):
  current_speed = speed.copy()
  current_pos = pos.copy()

  for _ in range(0, time_passed, dt):
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

@njit(fastmath=True)
def calc_sigma(r, a, theta):
  return r**2 + a**2+np.cos(theta)**2

@njit(fastmath=True)
def calc_delta(r, m, a):
  return r**2 - 2*m*r + a**2

@njit(fastmath=True)
def calc_phi_dot(delta, m, r, sigma, L, E, a, theta):
  return 1/delta * ((1 - (2*m*r)/sigma)*L + E*(2*m*r)/sigma*a*np.sin(theta)**2)

@njit(fastmath=True)
def calc_R(delta, C, kappa, r, L, E, a):
  return delta*(-C + kappa*r**2 - (L - a*E)**2) + (E*(r**2 + a**2) - L*a)**2

@njit(fastmath=True)
def calc_r_dot(sigma, R, is_prograde):
  factor = -1
  if is_prograde:
    factor = 1
  if R < 0:
    R = 0
  return factor * 1/sigma * np.sqrt(R)

@njit(fastmath=True)
def calc_uptheta(C, theta, kappa, E, L, a):
  return C + np.cos(theta)**2 * ((kappa + E**2)*a**2 - 1/np.sin(theta)**2 * L**2)

@njit(fastmath=True)
def calc_theta_dot(sigma, uptheta, is_prograde):
  factor = -1
  if is_prograde:
    factor = 1
  if uptheta < 0:
    uptheta = 0
  return factor * 1/sigma * np.sqrt(uptheta)

@njit(fastmath=True)
def step_relative(speed, pos, attractor_pos, mass, a, E, L, kappa, C, is_prograde, time_passed, dt=1):
  current_speed = speed.copy()
  current_pos = pos.copy()

  for _ in range(0, time_passed, dt):
    # r = np.sqrt(np.sum((current_pos-attractor_pos)**2))
    r = current_pos[0]
    sigma = calc_sigma(r, a, current_pos[2])
    delta = calc_delta(r, mass, a)
    R = calc_R(delta, C, kappa, r, L, E, a)
    uptheta = calc_uptheta(C, current_pos[2], kappa, E, L, a)
    # print(delta, R, sigma, uptheta, r)

    current_speed[0] = calc_r_dot(sigma, R, is_prograde)
    current_speed[1] = calc_phi_dot(delta, mass, r, sigma, L, E, a, current_pos[2])
    current_speed[2] = calc_theta_dot(sigma, uptheta, is_prograde)

    current_pos += current_speed * dt

  return current_speed, current_pos

class Relative_object:
  def __init__(self, attractor, pos, speed):
    # the pos and speed are in r, phi, theta, but keep theta=1/2*pi for now
    self.attractor = attractor
    self.pos = np.array(pos, dtype=float)
    self.speed = np.array(speed, dtype=float)
    self.E, self.L = self.calc_init_E_L()
    self.kappa = -1
    self.C = self.calc_C()

  def calc_init_E_L(self, is_prograde=True):
    # this is in an ideal scenario of theta=1/2*pi and an stable circular orbit
    # r = np.sqrt(np.sum((self.pos-self.attractor.pos)**2))
    r = self.pos[0]
    factor = -1
    if is_prograde:
      factor = 1

    # E = (r**2 - 2*self.attractor.mass*r + factor*self.attractor.a*np.sqrt(self.attractor.mass)) / (r * np.sqrt(r**2 - 3*self.attractor.mass*r + factor*2*self.attractor.a*np.sqrt(self.attractor.mass*r)))
    E = (r**2 - 2*self.attractor.mass*r + factor*self.attractor.a*np.sqrt(self.attractor.mass*r)) / (r * np.sqrt(r**2 - 3*self.attractor.mass*r + factor*2*self.attractor.a*np.sqrt(self.attractor.mass*r)))
    L = (factor*np.sqrt(self.attractor.mass)*(r**2 - factor*2*self.attractor.a*np.sqrt(self.attractor.mass*r) + self.attractor.a**2)) / (np.sqrt(r)*np.sqrt(r**2 - 3*self.attractor.mass*r + factor*2*self.attractor.a*np.sqrt(self.attractor.mass*r)))

    return E, L

  def calc_C(self):
    # r = np.sqrt(np.sum((self.pos-self.attractor.pos)**2))
    r = self.pos[0]
    sigma = calc_sigma(r, self.attractor.a, self.pos[2])
    return (sigma*self.speed[2])**2 - np.cos(self.pos[2])**2 * ((self.kappa + self.E**2)*self.attractor.a**2 - 1/np.sin(self.pos[2])**2 * self.L**2)

  def update(self, time_passed):
    # self.speed, self.pos = step_relative(self.speed, self.pos, self.attractor.pos, self.attractor.mass, time_passed)
    self.speed, self.pos = step_relative(self.speed, self.pos, self.attractor.pos, self.attractor.mass, self.attractor.a, self.E, self.L, self.kappa, self.C, True, time_passed)
    return (self.attractor.pos[0] + self.pos[0]*np.cos(self.pos[1]), self.attractor.pos[1] + self.pos[0]*np.sin(self.pos[1]))