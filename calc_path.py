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
  return r**2 + a**2*np.cos(theta)**2

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
def calc_R_tollerance(delta, C, kappa, r, L, E, a):
  # first calc the scale and then multiply by 1e-12 to adjust for the error of py floatingpoint calcs
  return 1e-12 * (abs(delta*(-C + kappa*r**2 - (L - a*E)**2)) + abs((E*(r**2 + a**2) - L*a)**2) + 1)

@njit(fastmath=True)
def calc_r_dot(sigma, R, is_prograde, radial_sign):
  factor = -1
  if is_prograde:
    factor = 1
  # print("R =", R)
  if R < 0:
    R = 0
  return radial_sign * 1/sigma * np.sqrt(R)

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
def calc_t_dot(delta, r, phi_dot, r_dot, kappa, mass, a):
  # ASSUMES EQUATORIAL

  # return 1/delta * (E*(r**2 + a**2 + (2*mass*r*a**2)/sigma * np.sin(theta)**2) * np.sin(theta)**2 - L * (2*mass*r)/sigma * a * np.sin(theta)**2)
  A = 1 - 2*mass/r
  B = 2*mass*a/r
  C = r**2 + a**2 + 2*mass*a**2/r
  # HERE C ISN'T CARTERS CONSTANT

  alpha = -A
  beta = -2*B*phi_dot
  gamma = C * phi_dot**2 + r**2/delta * r_dot**2 - kappa

  t1 = (-beta + np.sqrt(beta**2 - 4*alpha*gamma)) / (2*alpha)
  t2 = (-beta - np.sqrt(beta**2 - 4*alpha*gamma)) / (2*alpha)
  # only return the positive t_dot
  return max(t1, t2)

@njit(fastmath=True)
def calc_t_dot_from_E_L(delta, sigma, r, a, E, L, mass, theta):
  return 1/delta * (E * (r**2 + a**2 + 2*mass*r*a**2/sigma * np.sin(theta)**2) * np.sin(theta)**2 - L * 2*mass*r/sigma * a * np.sin(theta)**2)

# make solve_ivp handle when R<0
def radial_turning_event(tau, y, mass, a, E, L, C, kappa, radial_sign):
  r = y[0]
  delta = calc_delta(r, mass, a)
  return calc_R(delta, C, kappa, r, L, E, a)

radial_turning_event.terminal = True
radial_turning_event.direction = -1

# make solve_ivp stop when you want the result
def make_time_event(target_time):
  def time_event(tau, y, *args):
    return y[3] - target_time

  time_event.terminal = True
  time_event.direction = 1
  return time_event

# the differential part of the equ's, which needs to get integrated
def differential(tau, y, mass, a, E, L, C, kappa, radial_sign, is_prograde=True):
  r, phi, theta, t = y
  sigma = calc_sigma(r, a, theta)
  delta = calc_delta(r, mass, a)
  R = calc_R(delta, C, kappa, r, L, E, a)
  uptheta = calc_uptheta(C, theta, kappa, E, L, a)
  # print(delta, R, sigma, uptheta, r)

  # prevent errors, for when solve_ivp tries invalid values
  if R < 0.0:
    R = 0.0
  if uptheta < 0.0:
    uptheta = 0.0

  r_dot = calc_r_dot(sigma, R, is_prograde, radial_sign)
  phi_dot = calc_phi_dot(delta, mass, r, sigma, L, E, a, theta)
  theta_dot = calc_theta_dot(sigma, uptheta, is_prograde)
  t_dot = calc_t_dot_from_E_L(delta, sigma, r, a, E, L, mass, theta)

  return [r_dot, phi_dot, theta_dot, t_dot]
"""
def calc_final_speed(y, mass, a, E, L, C, kappa, radial_sign, is_prograde=True):
  r, phi, theta, t = y
  sigma = calc_sigma(r, a, theta)
  delta = calc_delta(r, mass, a)
  R = calc_R(delta, C, kappa, r, L, E, a)
  uptheta = calc_uptheta(C, theta, kappa, E, L, a)
  # print(delta, R, sigma, uptheta, r)

  # prevent errors, for when solve_ivp tries invalid values
  if R < 0.0:
    R = 0.0
  if uptheta < 0.0:
    uptheta = 0.0

  r_dot = calc_r_dot(sigma, R, is_prograde, radial_sign)
  phi_dot = calc_phi_dot(delta, mass, r, sigma, L, E, a, theta)
  theta_dot = calc_theta_dot(sigma, uptheta, is_prograde)

  return [r_dot, phi_dot, theta_dot]
"""
def step_relative_higher_accuracy_solve_ivp(pos, mass, a, E, L, kappa, C, radial_sign, time_passed, dt=1, ivp_solver_method="RK45", t_eval=[1]):
  # y is the pos array, used in the differential equ's
  y = np.array([pos[0], pos[1], pos[2], 0.0], dtype=float)
  if t_eval is None:
    t_targets = np.array([float(time_passed)])
  else:
    t_targets = np.asarray(t_eval, dtype=float)
  target_time = float(time_passed)
  # time_event = make_time_event(target_time)
  time_event = make_time_event(t_targets[-1])

  # start proper time "stopwatch" at 0
  tau = 0.0
  # init counter and stop
  k = 0
  n = t_targets.size
  # init output arrays
  pos_out = np.empty((n, 3))
  speed_out = np.empty((n, 3))

  # Use dt as the maximum step size, if provided.
  if dt is not None and dt > 0:
    max_step = float(dt)
  else:
    max_step = np.inf
  max_step = np.inf

  max_segments = 10000
  for _ in range(max_segments):
    # tau_end = tau + max(100.0, 10.0 * target_time + 100.0)
    tau_end = tau + max(100.0, 10.0 * t_targets[-1] + 1000.0)

    # dense output = True, makes it return an interpolating func
    sol = scipy.integrate.solve_ivp(differential, (tau, tau_end), y, args=(mass, a, E, L, C, kappa, radial_sign), method=ivp_solver_method, max_step=max_step, rtol=1e-10, atol=1e-12, events=[time_event, radial_turning_event], dense_output=True)
    if sol.status == -1:
      raise RuntimeError("solve_ivp failed: " + sol.message)

    # Latest state
    y = sol.y[:, -1]
    tau = sol.t[-1]

    t_seg_end = sol.y[3, -1]
    # sample every requested coordinate time inside this segment
    while k < n and t_targets[k] <= t_seg_end + 1e-9:
      tau_k = np.interp(t_targets[k], sol.y[3], sol.t)   # invert t(tau)
      yk = sol.sol(tau_k)                                # dense polynomial
      pos_out[k] = yk[:3]
      speed_out[k] = np.array(
        differential(tau_k, yk, mass, a, E, L, C, kappa, radial_sign))[:3]
      k += 1

    # check if the full time has passed
    if sol.t_events[0].size > 0:
      break
      """
      final_pos = y[:3].copy()
      final_t = float(y[3])

      # final_speed = calc_final_speed(final_pos, mass, a, E, L, kappa, C, radial_sign)
      final_speed = differential(tau, y, mass, a, E, L, kappa, C, radial_sign)

      # only return the speed of r, phi and theta
      return (final_speed[:3].copy(), final_pos.copy(), radial_sign, final_t)
      """

    # handle R<0
    if sol.t_events[1].size > 0:
      radial_sign *= -1.0
      y = sol.y[:, -1].copy()
      eps = 1e-8 * max(1.0, abs(y[0]))
      for d in (radial_sign, -radial_sign):
          r_test = y[0] + d * eps
          R_test = calc_R(calc_delta(r_test, mass, a), C, kappa, r_test, L, E, a)
          if R_test >= 0.0:
              y[0] = r_test
              break
      tau = sol.t[-1]
      continue

    # no event: extend the proper-time interval
    y = sol.y[:, -1].copy()
    tau = sol.t[-1]

    # If no event happened, extend the proper-time interval and continue.
    # continue
  if k < n:
    raise RuntimeError("Did not reach all requested snapshot times.")
  return speed_out, pos_out, radial_sign, t_targets
  raise RuntimeError("Too many radial turning-point segments.")

@njit(fastmath=True)
def step_relative_higher_accuracy(speed, pos, attractor_pos, mass, a, E, L, kappa, C, is_prograde, radial_sign, time_passed, dt=1, use_global_time=False):
  current_speed = speed.copy()
  current_pos = pos.copy()
  t_passed = 0
  previous_speed = speed.copy()
  previous_pos = pos.copy()
  previous_t = 0

  for _ in range(0, time_passed*max(1, int(1/dt)), 1):
    old_pos = current_pos.copy()
    if current_speed[1] > 0:
      is_prograde = True
    else:
      is_prograde = False
    # r = np.sqrt(np.sum((current_pos-attractor_pos)**2))
    r = current_pos[0]
    sigma = calc_sigma(r, a, current_pos[2])
    delta = calc_delta(r, mass, a)
    R = calc_R(delta, C, kappa, r, L, E, a)
    uptheta = calc_uptheta(C, current_pos[2], kappa, E, L, a)
    # print(delta, R, sigma, uptheta, r)

    current_speed[0] = calc_r_dot(sigma, R, is_prograde, radial_sign)
    current_speed[1] = calc_phi_dot(delta, mass, r, sigma, L, E, a, current_pos[2])
    current_speed[2] = calc_theta_dot(sigma, uptheta, is_prograde)

    current_pos += current_speed * dt

    # it is better to move the interpolation outside of this loop, to increase accuracy and just get returned 2 coords and 2 speed vectors, but the values in the object should not be changed, so it is more accurate, because there would be less rounding and estimation/interpolation errors
    if use_global_time:
      delta_t = calc_t_dot_from_E_L(delta, sigma, r, a, E, L, mass, pos[2]) * dt
      t_passed += delta_t

    if R >= 0.0:
      delta_new = calc_delta(current_pos[0], mass, a)
      R_new = calc_R(delta_new, C, kappa, current_pos[0], L, E, a)

      if R_new < 0.0:
        if R_new > -calc_R_tollerance(delta_new, C, kappa, current_pos[0], L, E, a):
          # handle rounding errors
          current_pos[0] = old_pos[0]
          current_speed[0] = 0.0
        else:
          current_pos = old_pos
          if use_global_time:
            t_passed -= delta_t
          radial_sign *= -1.0
    if t_passed >= time_passed:
      break
    if t_passed >= time_passed - delta_t*2:
      previous_speed = current_speed.copy()
      previous_pos = current_pos.copy()
      previous_t = t_passed
  return current_speed, current_pos, radial_sign, previous_speed, previous_pos, previous_t, t_passed

@njit(fastmath=True)
def step_relative(speed, pos, attractor_pos, mass, a, E, L, kappa, C, is_prograde, radial_sign, time_passed, dt=1, use_global_time=False):
  current_speed = speed.copy()
  current_pos = pos.copy()

  for _ in range(0, time_passed*max(1, int(1/dt)), 1):
    old_pos = current_pos.copy()
    if current_speed[1] > 0:
      is_prograde = True
    else:
      is_prograde = False
    # r = np.sqrt(np.sum((current_pos-attractor_pos)**2))
    r = current_pos[0]
    sigma = calc_sigma(r, a, current_pos[2])
    delta = calc_delta(r, mass, a)
    R = calc_R(delta, C, kappa, r, L, E, a)
    uptheta = calc_uptheta(C, current_pos[2], kappa, E, L, a)
    # print(delta, R, sigma, uptheta, r)

    current_speed[0] = calc_r_dot(sigma, R, is_prograde, radial_sign)
    current_speed[1] = calc_phi_dot(delta, mass, r, sigma, L, E, a, current_pos[2])
    current_speed[2] = calc_theta_dot(sigma, uptheta, is_prograde)

    current_pos += current_speed * dt

    if R >= 0.0:
      delta_new = calc_delta(current_pos[0], mass, a)
      R_new = calc_R(delta_new, C, kappa, current_pos[0], L, E, a)

      if R_new < 0.0:
        current_pos = old_pos
        radial_sign *= -1.0

  return current_speed, current_pos, radial_sign

class Relative_object:
  def __init__(self, attractor, pos, speed, stable_orbit=True):
    # the pos and speed are in r, phi, theta, but keep theta=1/2*pi for now
    self.attractor = attractor
    self.pos = np.array(pos, dtype=float)
    self.speed = np.array(speed, dtype=float)
    self.kappa = -1
    if stable_orbit:
      self.E, self.L = self.calc_init_E_L_stable_orbit()
    else:
      self.E, self.L = self.calc_init_E_L()
    self.C = self.calc_C()

    self.radial_sign = 1.0
    if self.speed[0] < 0.0:
      self.radial_sign = -1.0
    self.time = 0

  def calc_init_E_L(self, is_prograde=True):
    r = self.pos[0]
    # factor = -1
    # if is_prograde:
      # factor = 1

    sigma = calc_sigma(r, self.attractor.a, self.pos[2])
    delta = calc_delta(r, self.attractor.mass, self.attractor.a)
    # phi_dot = calc_phi_dot(delta, self.attractor.mass, r, sigma, self.L, self.E, self.attractor.a, self.pos[2])
    phi_dot = self.speed[1]
    t_dot = calc_t_dot(delta, r, phi_dot, self.speed[0], self.kappa, self.attractor.mass, self.attractor.a)
    # print("t_dot =", t_dot)
    # t_dot = 1
    E = (1 - (2*self.attractor.mass*r)/sigma)*t_dot + self.speed[1] * 2*self.attractor.mass*r/sigma * self.attractor.a * np.sin(self.pos[2])**2
    L = -t_dot * 2*self.attractor.mass*r/sigma * self.attractor.a * np.sin(self.pos[2])**2 + self.speed[1] * (r**2 + self.attractor.a**2 + 2*self.attractor.mass*r*self.attractor.a**2/sigma * np.sin(self.pos[2])**2) * np.sin(self.pos[2])**2

    return E, L

  def calc_init_E_L_stable_orbit(self, is_prograde=True):
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

  def update(self, time_passed, dt=1, use_global_time=False, ivp_solver_method=None, t_eval=None):
    # self.speed, self.pos = step_relative(self.speed, self.pos, self.attractor.pos, self.attractor.mass, time_passed)
    # if use_global_time:
      # self.speed, self.pos, self.radial_sign, previous_speed, previous_pos, previous_t, t_passed = step_relative_higher_accuracy(self.speed, self.pos, self.attractor.pos, self.attractor.mass, self.attractor.a, self.E, self.L, self.kappa, self.C, True, self.radial_sign, time_passed, dt, True)
    if ivp_solver_method:
      # final_speed[:3], final_pos, radial_sign, final_t
      (self.speed, self.pos, self.radial_sign, t_passed) = step_relative_higher_accuracy_solve_ivp(self.pos, self.attractor.mass, self.attractor.a, self.E, self.L, self.kappa, self.C, self.radial_sign, time_passed, dt, ivp_solver_method, t_eval)
      self.time += t_passed
      return (self.attractor.pos[0] + self.pos[0]*np.cos(self.pos[1]), self.attractor.pos[1] + self.pos[0]*np.sin(self.pos[1]))
    if use_global_time:
      (self.speed, self.pos, self.radial_sign, prev_speed, prev_pos, prev_t, t_passed) = step_relative_higher_accuracy(self.speed, self.pos, self.attractor.pos, self.attractor.mass, self.attractor.a, self.E, self.L, self.kappa, self.C, True, self.radial_sign, time_passed, dt, True)

      span = t_passed - prev_t
      frac = (time_passed - prev_t)/span if span > 0.0 else 1.0
      frac = min(max(frac, 0.0), 1.0)

      self.interp_pos = prev_pos + frac*(self.pos - prev_pos)   # state at EXACTLY time_passed

      # speed consistent with the landed position (don't lerp prev_speed: it's one step stale)
      r_i, th_i = self.interp_pos[0], self.interp_pos[2]
      sigma_i = calc_sigma(r_i, self.attractor.a, th_i)
      delta_i = calc_delta(r_i, self.attractor.mass, self.attractor.a)
      R_i  = calc_R(delta_i, self.C, self.kappa, r_i, self.L, self.E, self.attractor.a)
      up_i = calc_uptheta(self.C, th_i, self.kappa, self.E, self.L, self.attractor.a)
      self.interp_speed = np.empty(3)
      self.interp_speed[0] = calc_r_dot(sigma_i, R_i, True, self.radial_sign)
      self.interp_speed[1] = calc_phi_dot(delta_i, self.attractor.mass, r_i, sigma_i,
                                          self.L, self.E, self.attractor.a, th_i)
      self.interp_speed[2] = calc_theta_dot(sigma_i, up_i, True)

      self.time += time_passed          # exact, shared render clock
      p = self.interp_pos
      return (self.attractor.pos[0] + p[0]*np.cos(p[1]), self.attractor.pos[1] + p[0]*np.sin(p[1]))
    else:
      self.speed, self.pos, self.radial_sign = step_relative(self.speed, self.pos, self.attractor.pos, self.attractor.mass, self.attractor.a, self.E, self.L, self.kappa, self.C, True, self.radial_sign, time_passed, dt)
    
    return (self.attractor.pos[0] + self.pos[0]*np.cos(self.pos[1]), self.attractor.pos[1] + self.pos[0]*np.sin(self.pos[1]))