from calc_path import *
import numpy as np
import time
import scipy
# for the multithreading(only got 2x speed increase, from ~30 sec to 15 sec)
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

time_start = time.time()

# init objects
w, h = 1280, 720

attractor_mass = 1
attractors = [Attractor(attractor_mass, (w/2, h/2), a/100, convert_a=True) for a in range(-99, 100)]
orbiters = [Relative_object(attractor, (200, 0, np.pi/2), (-0.000002, 0.0004, 0), False) for attractor in attractors]
# orbiters = [Relative_object(attractor, (200, 0, np.pi/2), (0, 0, 0), True) for attractor in attractors]
orbiters_cache = []

for idx in [0, 99, -1]:
  orb = orbiters[idx]

  a = orb.attractor.a
  M = orb.attractor.mass
  r = orb.pos[0]
  theta = orb.pos[2]

  sigma = calc_sigma(r, a, theta)
  delta = calc_delta(r, M, a)

  phi_dot = calc_phi_dot(delta, M, r, sigma, orb.L, orb.E, a, theta)
  t_dot = calc_t_dot_from_E_L(delta, sigma, r, a, orb.E, orb.L, M, theta)

  omega = phi_dot / t_dot

  print("index:", idx)
  print("a:", a)
  print("E:", orb.E)
  print("L:", orb.L)
  print("phi_dot:", phi_dot)
  print("t_dot:", t_dot)
  print("omega = dphi/dt:", omega)
  print()
test_orbiters = [
    Relative_object(attractors[0], (200, 0, np.pi/2), (0.0, 0.0, 0.0), True),
    Relative_object(attractors[99], (200, 0, np.pi/2), (0.0, 0.0, 0.0), True),
    Relative_object(attractors[-1], (200, 0, np.pi/2), (0.0, 0.0, 0.0), True),
]

for i, orb in enumerate(test_orbiters):
    orb.update(100, 0.05, True)
    print(i, orb.attractor.a, orb.interp_pos[1])

def kappa_check(pos, speed, M, a, E, L):
  r, phi, theta = pos
  r_dot, phi_dot, theta_dot = speed
  sigma = r**2 + a**2*math.cos(theta)**2
  delta = r**2 - 2*M*r + a**2
  s2 = math.sin(theta)**2
  t_dot = calc_t_dot_from_E_L(delta, sigma, r, a, E, L, M, theta)
  g_tt = -(1 - 2*M*r/sigma); g_tp = -2*M*r/sigma*a*s2
  g_rr = sigma/delta; g_th = sigma
  g_pp = (r**2 + a**2 + 2*M*r*a**2/sigma*s2)*s2
  return (g_tt*t_dot**2 + 2*g_tp*t_dot*phi_dot + g_rr*r_dot**2 + g_th*theta_dot**2 + g_pp*phi_dot**2)    # should be ≈ -1
"""
time_start = time.time()
# gen data

t_eval = np.arange(100.0, 100000.0 + 1.0, 100.0)
for orbiter in orbiters:
  # current_cache = []
  orbiter.update(100, 0.05, ivp_solver_method="RK45", t_eval=t_eval)
  current_cache = [[orbiter.pos[i].copy(), orbiter.speed[i].copy()] for i in range(len(t_eval))]
  
  # for i in range(1000):
    # orbiter.update(100, 0.05)
    # current_cache.append([orbiter.pos.copy(), orbiter.speed.copy()])
    # orbiter.update(100, 0.05, True)
    # current_cache.append([orbiter.interp_pos.copy(), orbiter.interp_speed.copy()])
    # t_eval = np.arange(100.0, 100000.0 + 1.0, 100.0)
    # orbiter.update(100, 0.05, ivp_solver_method="RK45", t_eval=t_eval)
    # current_cache.append([orbiter.pos.copy(), orbiter.speed.copy()])
    # print(f'i = {i}, {time.time()-time_start}')
    
  
  orbiters_cache.append(current_cache)
  kappa = kappa_check(current_cache[-1][0], current_cache[-1][1], orbiter.attractor.mass, orbiter.attractor.a, orbiter.E, orbiter.L)
  print("finished orbiter with a =", orbiter.attractor.a, "kappa = ", kappa)

print(f'finished orbiter calc in {time.time()-time_start} seconds')
"""
t_eval = np.arange(100.0, 100000.0 + 1.0, 100.0)
def calculate_single_orbiter(orbiter):
  orbiter.update(100, 0.05, ivp_solver_method="RK45", t_eval=t_eval)
  current_cache = [[orbiter.pos[i].copy(), orbiter.speed[i].copy()] for i in range(len(t_eval))]
  orbiters_cache.append(current_cache)
  kappa = kappa_check(current_cache[-1][0], current_cache[-1][1], orbiter.attractor.mass, orbiter.attractor.a, orbiter.E, orbiter.L)
  # print("finished orbiter with a =", orbiter.attractor.a, "kappa = ", kappa)

  return current_cache, orbiter.attractor.a, kappa, orbiter.tau_all, orbiter.t_dot_all

# gen data in paralel
if __name__ == "__main__":
  time_start = time.time()
  # Use Pool() to automatically use all available CPU cores
  with Pool() as pool:
    # pool.map runs calculate_single_orbiter on each item in parallel
    results = pool.map(calculate_single_orbiter, orbiters)
  orbiters_cache = []
  tau_cache = []
  t_dot_cache = []
  for current_cache, attractor_a, kappa, tau_all, t_dot in results:
    orbiters_cache.append(current_cache)
    tau_cache.append(tau_all)
    t_dot_cache.append(t_dot)
    assert round(abs(kappa+1), 14) == 0, "kappa isn't close to -1, so the results aren't valid"
    print("finished orbiter with a =", attractor_a, "kappa =", kappa)
  print(f'finished orbiter calc in {time.time()-time_start} seconds')

  time_start = time.time()

  # calc delta's
  deltas = []
  if False:
    for i in range(len(orbiters_cache)):
      orbiter_deltas = []
      for j in range(i+1, len(orbiters_cache)):
        orbiter_deltas.append([[orbiters_cache[i][x][y] - orbiters_cache[j][x][y] for y in range(2)] for x in range(len(orbiters_cache[j]))])
      deltas.append(orbiter_deltas)

  print("deltas_finished")
  print(f'finished delta calc in {time.time()-time_start} seconds')

  def lookup_deltas(i, j, snapshot_index):
    if i < 0:
      i = len(orbiters_cache) + i
    if j < 0:
        j = len(orbiters_cache) + j
    # returns pos, speed
    assert i != j
    if i > j:
      i, j = j, i

    a1 = -0.99 + 0.01*i
    a2 = -0.99 + 0.01*j
    d1 = calc_proper_distance(orbiters_cache[i][snapshot_index][0], orbiters_cache[j][snapshot_index][0], attractor_mass, a1)
    d2 = calc_proper_distance(orbiters_cache[i][snapshot_index][0], orbiters_cache[j][snapshot_index][0], attractor_mass, a2)
    d = (d1 + d2)/2
    err = abs(d1 - d2)

    d_tau = tau_cache[j][snapshot_index] - tau_cache[i][snapshot_index]
    d_t_dot = t_dot_cache[j][snapshot_index] - t_dot_cache[i][snapshot_index]
    avg_t_dot = (t_dot_cache[i][snapshot_index] + t_dot_cache[j][snapshot_index])/2

    if len(deltas) > 0:
      return deltas[i][j-i-1][snapshot_index], d, err
    else:
      # return [orbiters_cache[i][snapshot_index][0] - orbiters_cache[j][snapshot_index][0], orbiters_cache[i][snapshot_index][1] - orbiters_cache[j][snapshot_index][1]], d, err, float(d_tau), float(d_t_dot), float(t_dot_cache[i][snapshot_index]), float(t_dot_cache[j][snapshot_index]), float(min(t_dot_cache[j])), float(max(t_dot_cache[j]))
      return [orbiters_cache[i][snapshot_index][0] - orbiters_cache[j][snapshot_index][0], orbiters_cache[i][snapshot_index][1] - orbiters_cache[j][snapshot_index][1]], d, err, float(d_tau), float(d_t_dot)

  def calc_proper_distance(p1, p2, M, a):
    # since the coords were at the same time, dt=0
    r, phi, theta = (p1 + p2)/2
    rho_2 = r**2 + a**2*math.cos(theta)**2
    delta = r**2 - 2*M*r + a**2
    d = p1-p2
    ds_2 =  rho_2/delta*d[0]**2 + rho_2*d[2]**2 + (r**2 + a**2)*math.sin(theta)**2*d[1]**2 + 2*M*r/rho_2*(a*math.sin(theta)**2*d[1])**2
    return math.sqrt(ds_2)

  print(lookup_deltas(0, 99, 999))
  print(lookup_deltas(99, -1, 999))
  print(lookup_deltas(0, -1, 999), '\n')

  print(lookup_deltas(0, 99, 0))
  print(lookup_deltas(99, -1, 0))
  print(lookup_deltas(0, -1, 0), '\n')

  print(lookup_deltas(0, 99, 1))
  print(lookup_deltas(99, -1, 1))
  print(lookup_deltas(0, -1, 1))

  def get_plot_points(i, j, end_time_index):
    ds_list = np.empty(end_time_index)
    dtau_list = np.empty(end_time_index)
    for time_index in range(end_time_index):
      current_deltas, ds, err, dtau, d_t_dot = lookup_deltas(i, j, time_index)
      ds_list[time_index] = ds
      dtau_list[time_index] = dtau
    return ds_list, dtau_list

  # plot the differences, with respect to t
  fig, (ax_ds, ax_dtau) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
  ax_ds.set_ylabel("ds(light s)")
  ax_dtau.set_ylabel("dtau(s)")
  ax_dtau.set_xlabel("t(s)")
  fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.32, hspace=0.15)
  ax_t  = plt.axes([0.10, 0.20, 0.86, 0.03])
  ax_s1 = plt.axes([0.10, 0.13, 0.86, 0.03])
  ax_s2 = plt.axes([0.10, 0.06, 0.86, 0.03])
  s_t  = Slider(ax_t,  "t max", t_eval[1], t_eval[-1], valinit=t_eval[-1], valstep=100)
  s_s1 = Slider(ax_s1, "a1",   -0.99, 0.99, valinit=0, valstep=0.01)
  s_s2 = Slider(ax_s2, "a2",   -0.99, 0.99, valinit= 0.99, valstep=0.01)

  i0, j0 = 99, 198
  ds_full, dtau_full = get_plot_points(i0, j0, len(t_eval))

  line_ds,   = ax_ds.plot(t_eval, ds_full)
  line_dtau, = ax_dtau.plot(t_eval, dtau_full)

  state = {"i": i0, "j": j0}
  pair_cache = {(i0, j0): (ds_full, dtau_full)}

  def idx_of_a(a):
    return int(round((a + 0.99) / 0.01))

  def get_pair(i, j):
    key = (min(i, j), max(i, j))
    if key not in pair_cache:
      pair_cache[key] = get_plot_points(key[0], key[1], len(t_eval))
    return pair_cache[key]

  def redraw():
    n = int(s_t.val / 100)
    ds, dtau = get_pair(state["i"], state["j"])
    line_ds.set_data(t_eval[:n], ds[:n])
    line_dtau.set_data(t_eval[:n], dtau[:n])
    for ax in (ax_ds, ax_dtau):
      ax.relim()
      ax.autoscale_view()
    ax_ds.set_title(
        f"a1 = {-0.99 + 0.01*state['i']:+.2f},  a2 = {-0.99 + 0.01*state['j']:+.2f}"
        f"   |   t max = {int(s_t.val)} s"
        f"   |   ds = {ds[n-1]:.4g},  dtau = {dtau[n-1]:.4g}")
    fig.canvas.draw_idle()

  def on_spin(_):
    i, j = idx_of_a(s_s1.val), idx_of_a(s_s2.val)
    if i == j:
      return
    state["i"], state["j"] = i, j
    redraw()

  s_t.on_changed(lambda _: redraw())
  s_s1.on_changed(on_spin)
  s_s2.on_changed(on_spin)

  plt.show()
