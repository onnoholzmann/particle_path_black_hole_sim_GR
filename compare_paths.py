from calc_path import *
import numpy as np
import time
import scipy
from scipy.signal import find_peaks
# for the multithreading(only got 2x speed increase, from ~30 sec to 15 sec)
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from numba import njit

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
t_eval = np.arange(100.0, 100000.0 + 1.0, 10.0)
def calculate_single_orbiter(orbiter):
  orbiter.update(10, 0.05, ivp_solver_method="RK45", t_eval=t_eval)
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

  def calc_proper_distance_line_segment(p1, p2, M, a):
    # since the coords were at the same time, dt=0
    r, phi, theta = (p1 + p2)/2
    rho_2 = r**2 + a**2*math.cos(theta)**2
    delta = r**2 - 2*M*r + a**2
    d = p1-p2
    dphi = (d[1] + math.pi) % (2*math.pi) - math.pi
    ds_2 =  rho_2/delta*d[0]**2 + rho_2*d[2]**2 + (r**2 + a**2)*math.sin(theta)**2*dphi**2 + 2*M*r/rho_2*(a*math.sin(theta)**2*dphi)**2
    # ds_2 =  rho_2/delta*d[0]**2 + rho_2*d[2]**2 + (r**2 + a**2)*math.sin(theta)**2*d[1]**2 + 2*M*r/rho_2*(a*math.sin(theta)**2*d[1])**2
    return math.sqrt(ds_2)

  @njit(fastmath=True)
  def ds_integrand(lam, p1, p2, M, a):
    # since the coords were at the same time, dt=0
    r, phi, theta = p1 + lam*(p2 - p1)
    rho_2 = r**2 + a**2*math.cos(theta)**2
    delta = r**2 - 2*M*r + a**2
    d = p2 - p1
    dphi = (d[1] + math.pi) % (2*math.pi) - math.pi
    ds_2 =  rho_2/delta*d[0]**2 + rho_2*d[2]**2 + (r**2 + a**2)*math.sin(theta)**2*dphi**2 + 2*M*r/rho_2*(a*math.sin(theta)**2*dphi)**2
    return math.sqrt(ds_2)

  def calc_proper_distance(p1, p2, M, a):
    # quad returns (value, error), so take [0]
    return scipy.integrate.quad(ds_integrand, 0, 1, args=(p1, p2, M, a), epsabs=1e-12, epsrel=1e-12)[0]

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
    n = int(np.searchsorted(t_eval, s_t.val, side="right"))
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

  def calc_precession(r, phi, t=None, use_minima=True,
                    distance=None, prominence_factor=0.25, refine=True):
    """
    r, phi : orbit arrays (phi unwrapped, radians)
    t      : optional coordinate-time array (for sub-sample refinement)

    returns: peaks, phi_at_turning_points, precession_per_radial_period
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # radial turning points (periapses if use_minima, else apoapses)
    signal = -r if use_minima else r

    if distance is None:
        distance = max(2, len(r) // 50)

    prominence = prominence_factor * (np.max(r) - np.min(r))
    if prominence <= 0:
        prominence = 1e-12

    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)

    # optional sub-sample refinement of each turning point (parabola through 3 points)
    if refine and t is not None:
        t = np.asarray(t, dtype=float)
        phi_turn = []
        for p in peaks:
            if 0 < p < len(r) - 1:
                y0, y1, y2 = r[p-1], r[p], r[p+1]
                denom = y0 - 2*y1 + y2
                if abs(denom) > 1e-30:
                    delta = 0.5 * (y0 - y2) / denom        # offset in samples
                    t_turn = t[p] + delta * (t[p+1] - t[p])
                    phi_turn.append(np.interp(t_turn, t, phi))
                else:
                    phi_turn.append(phi[p])
            else:
                phi_turn.append(phi[p])
        phi_turn = np.array(phi_turn)
    else:
        phi_turn = phi[peaks]

    # angle advanced between successive turning points
    dphi = np.diff(phi_turn)

    # a closed (Newtonian) orbit would give exactly 2*pi per period
    precession = dphi - 2*np.pi

    return peaks, phi_turn, precession

  def calc_rotations(phi):
    """total azimuthal revolutions around the centre"""
    phi = np.asarray(phi, dtype=float)
    return (phi[-1] - phi[0]) / (2*np.pi)

  # the precession orbit graph
  r_vals = [state[0][0] for state in orbiters_cache[-1]]
  phi_vals = [state[0][1] for state in orbiters_cache[-1]]

  phi_wrapped = [p % (2 * np.pi) for p in phi_vals]  # optional: wrap to [0, 2π]

  fig_prec = plt.figure(figsize=(12, 14))
  gs = GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1], hspace=0.35)
  ax_prec = fig_prec.add_subplot(gs[:, 0], projection="polar")
  ax_tdot = fig_prec.add_subplot(gs[0, 1])
  ax_r = fig_prec.add_subplot(gs[1, 1])
  ax_omega = fig_prec.add_subplot(gs[2, 1])
  fig_prec.subplots_adjust(bottom=0.22)

  locked_scale = True
  if locked_scale:
     # lock t_dot scale
    all_tdot = np.concatenate([np.asarray(arr, dtype=float) for arr in t_dot_cache])

    tdot_min = np.min(all_tdot)
    tdot_max = np.max(all_tdot)

    tdot_span = tdot_max - tdot_min
    if tdot_span > 0:
        tdot_pad = 0.05 * tdot_span
    else:
        tdot_pad = 0.01

    ax_tdot.set_xlim(t_eval[0], t_eval[-1])
    ax_tdot.set_ylim(tdot_min - tdot_pad, tdot_max + tdot_pad)
    ax_tdot.autoscale(False)

    # lock polar radius scale
    global_r_max = max(entry[0][0] for cache in orbiters_cache for entry in cache)

    ax_prec.set_rmin(0)
    ax_prec.set_rmax(global_r_max * 1.05)
    ax_prec.autoscale(False)

        # lock radius graph scale
    all_r = np.concatenate([
        np.array([entry[0][0] for entry in cache], dtype=float)
        for cache in orbiters_cache
    ])

    r_min = np.min(all_r)
    r_max = np.max(all_r)

    r_span = r_max - r_min
    if r_span > 0:
        r_pad = 0.05 * r_span
    else:
        r_pad = 0.01

    ax_r.set_xlim(t_eval[0], t_eval[-1])
    ax_r.set_ylim(r_min - r_pad, r_max + r_pad)
    ax_r.autoscale(False)

    # lock omega graph scale
    all_omega = np.concatenate([
        np.array([entry[1][1] for entry in cache], dtype=float) /
        np.asarray(t_dot_cache[cache_idx], dtype=float)
        for cache_idx, cache in enumerate(orbiters_cache)
    ])

    omega_min = np.min(all_omega)
    omega_max = np.max(all_omega)

    omega_span = omega_max - omega_min
    if omega_span > 0:
        omega_pad = 0.05 * omega_span
    else:
        omega_pad = 0.01

    ax_omega.set_xlim(t_eval[0], t_eval[-1])
    ax_omega.set_ylim(omega_min - omega_pad, omega_max + omega_pad)
    ax_omega.autoscale(False)

  a_init = 0.0
  idx_a0 = idx_of_a(a_init)
  cache0 = orbiters_cache[idx_a0]
  phi0 = np.array([entry[0][1] for entry in cache0], dtype=float)
  r0 = np.array([entry[0][0] for entry in cache0], dtype=float)
  tdot0 = np.asarray(t_dot_cache[idx_a0], dtype=float)
  omega0 = np.array([entry[1][1] for entry in cache0], dtype=float) / tdot0

  # stores whether the animation is currently playing
  anim = {"playing": False, "step": 1000, "playing_a": False, "step_a": 0.03, "direction_a": 1, "pause_ticks_a": 0}

  # start with the full graph visible, because the default time is the end
  line_prec, = ax_prec.plot(phi0 % (2 * np.pi), r0, linewidth=0.7, alpha=0.85)

  # t_dot graph under the precession graph
  line_tdot, = ax_tdot.plot(t_eval, tdot0, linewidth=1.0)
  ax_tdot.set_xlabel("t (s)")
  ax_tdot.set_ylabel("t_dot = dt/dtau")
  ax_tdot.grid(True)

  # radius graph
  line_r, = ax_r.plot(t_eval, r0, linewidth=1.0)
  ax_r.set_ylabel("r")
  ax_omega.set_xlabel("t (s)")
  ax_r.grid(True)

  # angular velocity graph
  line_omega, = ax_omega.plot(t_eval, omega0, linewidth=1.0)
  ax_omega.set_ylabel("omega = dphi/dt")
  ax_omega.set_xlabel("t (s)")
  ax_omega.grid(True)

  ax_a = plt.axes([0.15, 0.13, 0.70, 0.03])
  s_a  = Slider(ax_a, "a", -0.99, 0.99, valinit=a_init, valstep=0.01)

  ax_time = plt.axes([0.15, 0.08, 0.70, 0.03])
  s_time  = Slider(ax_time, "t", t_eval[0], t_eval[-1], valinit=t_eval[-1], valstep=100)

  ax_play = plt.axes([0.15, 0.03, 0.08, 0.04])
  b_play = Button(ax_play, "Play")
  ax_play_a = plt.axes([0.25, 0.03, 0.10, 0.04])
  b_play_a = Button(ax_play_a, "Play a")
  precession_cache = {idx_a0: (phi0, r0, tdot0, omega0)}

  def get_precession_data(idx):
    idx = max(0, min(idx, len(orbiters_cache) - 1))

    if idx not in precession_cache:
        cache = orbiters_cache[idx]
        phi = np.array([entry[0][1] for entry in cache], dtype=float)
        r   = np.array([entry[0][0] for entry in cache], dtype=float)
        tdot = np.asarray(t_dot_cache[idx], dtype=float)
        omega = np.array([entry[1][1] for entry in cache], dtype=float) / tdot
        precession_cache[idx] = (phi, r, tdot, omega)

    return precession_cache[idx], idx

  def set_axes_for_a(idx):
    if not locked_scale:
      (phi, r, tdot, omega), idx = get_precession_data(idx)

      # polar plot limits
      ax_prec.set_rmin(0)
      ax_prec.set_rmax(np.max(r) * 1.05)

      # t_dot plot limits
      ax_tdot.set_xlim(t_eval[0], t_eval[-1])

      ymin = np.min(tdot)
      ymax = np.max(tdot)
      pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01

      ax_tdot.set_ylim(ymin - pad, ymax + pad)

      # radius plot limits
      ax_r.set_xlim(t_eval[0], t_eval[-1])

      ymin = np.min(r)
      ymax = np.max(r)
      pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01

      ax_r.set_ylim(ymin - pad, ymax + pad)

      # omega plot limits
      ax_omega.set_xlim(t_eval[0], t_eval[-1])

      ymin = np.min(omega)
      ymax = np.max(omega)
      pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01

      ax_omega.set_ylim(ymin - pad, ymax + pad)

  a_vals = np.array([-0.99 + 0.01 * i for i in range(len(orbiters_cache))])
  
  n_orb = len(orbiters_cache)
  tau_final     = np.empty(n_orb)
  avg_tdot      = np.empty(n_orb)
  min_tdot      = np.empty(n_orb)
  max_tdot      = np.empty(n_orb)
  prec_mean_deg = np.full(n_orb, np.nan)
  loops_total   = np.empty(n_orb)
  n_periods     = np.zeros(n_orb, dtype=int)
  
  for i, cache in enumerate(orbiters_cache):
    r_arr   = np.array([e[0][0] for e in cache], dtype=float)
    phi_arr = np.array([e[0][1] for e in cache], dtype=float)
    tdot_arr = np.asarray(t_dot_cache[i], dtype=float)
  
    # 1. proper time at the final coordinate time
    tau_final[i] = tau_cache[i][-1]
  
    # 2. time-dilation factors (avg, min, max)
    avg_tdot[i] = np.mean(tdot_arr)
    min_tdot[i] = np.min(tdot_arr)
    max_tdot[i] = np.max(tdot_arr)
  
    # 4. total azimuthal loops
    loops_total[i] = calc_rotations(phi_arr)
  
    # 3. mean precession per radial period
    peaks, phi_turn, prec = calc_precession(r_arr, phi_arr, t=t_eval)
    n_periods[i] = len(prec)
    if len(prec) > 0:
      prec_mean_deg[i] = np.degrees(np.mean(prec))

  def redraw_precession(_=None):
    idx = idx_of_a(s_a.val)
    (phi, r, tdot, omega), idx = get_precession_data(idx)

    t_val = float(s_time.val)

    # convert time to number of snapshots
    n = int(np.searchsorted(t_eval, t_val, side="right"))
    n = max(1, min(n, len(t_eval), len(phi), len(tdot)))

    frame = n - 1

    # update only up to the current time
    line_prec.set_data(phi[:n] % (2 * np.pi), r[:n])
    line_tdot.set_data(t_eval[:n], tdot[:n])

    line_r.set_data(t_eval[:n], r[:n])
    line_omega.set_data(t_eval[:n], omega[:n])

    a_val = -0.99 + 0.01 * idx
    t_current = t_eval[frame]

    ax_prec.set_title(
        f"Perihelion Precession: a = {a_val:+.2f}, t = {t_current:.0f} s"
    )
    if np.isnan(prec_mean_deg[idx]):
        prec_str = "n/a (no clear turning points)"
    else:
        prec_str = f"{prec_mean_deg[idx]:.3f} deg/loop"

    ax_prec.set_title(
        f"Perihelion Precession: a = {a_val:+.2f}, t = {t_current:.0f} s\n"
        f"precession/loop: {prec_str}   |   total loops: {loops_total[idx]:.2f}"
        f"   |   radial periods: {n_periods[idx]}",
        fontsize=10,
    )

    fig_prec.canvas.draw_idle()

  def set_a_programmatic(a_val):
    # Clamp to valid index
    idx = idx_of_a(a_val)
    idx = max(0, min(idx, len(orbiters_cache) - 1))

    # Snap to exact a-value belonging to that index
    a_snapped = -0.99 + 0.01 * idx

    # Change slider without triggering its callback
    s_a.eventson = False
    s_a.set_val(a_snapped)
    s_a.eventson = True

    # While automatically scanning a, keep time at the end
    s_time.eventson = False
    s_time.set_val(t_eval[-1])
    s_time.eventson = True
    set_axes_for_a(idx)
    redraw_precession()

  def tick():
      if not anim["playing"]:
          return

      new_time = float(s_time.val) + anim["step"]

      # loop back to the beginning
      if new_time > t_eval[-1]:
          new_time = t_eval[0]

      # move the time slider without triggering its callback twice
      s_time.eventson = False
      s_time.set_val(new_time)
      s_time.eventson = True

      redraw_precession()

  def tick_a():
      if not anim["playing_a"]:
          return

      # optional pause at the endpoints
      if anim["pause_ticks_a"] > 0:
          anim["pause_ticks_a"] -= 1
          return

      new_a = float(s_a.val) + anim["direction_a"] * anim["step_a"]

      # bounce at the right end
      if new_a >= 0.99:
          new_a = 0.99
          anim["direction_a"] = -1

          # pause for a few timer ticks at the endpoint
          anim["pause_ticks_a"] = 10

      # bounce at the left end
      elif new_a <= -0.99:
          new_a = -0.99
          anim["direction_a"] = 1

          # pause for a few timer ticks at the endpoint
          anim["pause_ticks_a"] = 10

      set_a_programmatic(new_a)

  timer = fig_prec.canvas.new_timer(interval=20)
  timer.add_callback(tick)

  timer_a = fig_prec.canvas.new_timer(interval=20)
  timer_a.add_callback(tick_a)

  def toggle_play(_):
      # t and a never run together: stop a-animation if it is active
      if anim["playing_a"]:
          anim["playing_a"] = False
          b_play_a.label.set_text("Play a")
          timer_a.stop()

      anim["playing"] = not anim["playing"]

      b_play.label.set_text("Pause" if anim["playing"] else "Play")

      if anim["playing"]:
          # if already at the end, restart from the beginning immediately
          if float(s_time.val) >= t_eval[-1]:
              s_time.eventson = False
              s_time.set_val(t_eval[0])
              s_time.eventson = True
              redraw_precession()

          timer.start()
      else:
          timer.stop()

      fig_prec.canvas.draw_idle()

  def toggle_play_a(_):
      # t and a never run together: stop t-animation if it is active
      if anim["playing"]:
          anim["playing"] = False
          b_play.label.set_text("Play")
          timer.stop()

      anim["playing_a"] = not anim["playing_a"]

      b_play_a.label.set_text("Pause a" if anim["playing_a"] else "Play a")

      if anim["playing_a"]:
          # if already at maximum a, restart from minimum a
          if float(s_a.val) >= 0.99:
              set_a_programmatic(-0.99)

          timer_a.start()
      else:
          timer_a.stop()

      fig_prec.canvas.draw_idle()

  def on_a_change(_):
      # manual a slider change stops both animations
      anim["playing"] = False
      anim["playing_a"] = False

      b_play.label.set_text("Play")
      b_play_a.label.set_text("Play a")

      timer.stop()
      timer_a.stop()

      # reset bounce direction
      anim["direction_a"] = -1 if float(s_a.val) >= 0.99 else 1
      anim["pause_ticks_a"] = 0

      # reset time to the end, not the beginning
      s_time.eventson = False
      s_time.set_val(t_eval[-1])
      s_time.eventson = True

      set_axes_for_a(idx_of_a(s_a.val))
      redraw_precession()

  def on_time_change(_):
      # manually changing time stops the a-animation
      if anim["playing_a"]:
          anim["playing_a"] = False
          b_play_a.label.set_text("Play a")
          timer_a.stop()

      redraw_precession()

  s_a.on_changed(on_a_change)
  s_time.on_changed(on_time_change)
  b_play.on_clicked(toggle_play)
  b_play_a.on_clicked(toggle_play_a)

  # initial axis limits and initial draw
  set_axes_for_a(idx_a0)
  redraw_precession()

  # keep widget references alive
  fig_prec._widgets = (s_a, s_time, b_play, b_play_a, timer, timer_a)

  fig_a = plt.figure(figsize=(12, 9))
  gs_a = GridSpec(2, 2, hspace=0.4, wspace=0.25)

  # 1. tau at the final t
  ax_tau = fig_a.add_subplot(gs_a[0, 0])
  ax_tau.plot(a_vals, tau_final, lw=1.2)
  ax_tau.axvline(0.0, color="gray", lw=0.8, ls="--")
  ax_tau.set_xlabel("a")
  ax_tau.set_ylabel("tau at final t (s)")
  ax_tau.set_title("Proper time elapsed at final coordinate time")
  ax_tau.grid(True)

  # 2. time-dilation factors (avg, min, max)
  ax_tdot = fig_a.add_subplot(gs_a[0, 1])
  
  ax_tdot.plot(a_vals, avg_tdot, lw=1.5, color='blue', label='Average')
  ax_tdot.plot(a_vals, min_tdot, lw=1.0, color='green', ls='--', label='Minimum (apoapsis)')
  ax_tdot.plot(a_vals, max_tdot, lw=1.0, color='red', ls='--', label='Maximum (periapsis)')
  
  ax_tdot.axvline(0.0, color="gray", lw=0.8, ls="--")
  ax_tdot.set_xlabel("a")
  ax_tdot.set_ylabel("t_dot = dt/dtau")
  ax_tdot.set_title("Time-dilation factor (min, avg, max)")
  ax_tdot.grid(True)
  ax_tdot.legend(fontsize=8, loc='best')

  # 3. precession per radial period
  ax_prec_a = fig_a.add_subplot(gs_a[1, 0])
  ax_prec_a.plot(a_vals, prec_mean_deg, lw=1.2, marker=".", ms=3)
  ax_prec_a.axvline(0.0, color="gray", lw=0.8, ls="--")
  ax_prec_a.set_xlabel("a")
  ax_prec_a.set_ylabel("mean precession per radial period (deg)")
  ax_prec_a.set_title("Periapsis precession per orbit")
  ax_prec_a.grid(True)

  # 4. amount of loops
  ax_loops = fig_a.add_subplot(gs_a[1, 1])
  ax_loops.plot(a_vals, loops_total, lw=1.2, label="azimuthal loops = dphi/2pi")
  ax_loops.plot(a_vals, n_periods, lw=1.0, ls="--", label="radial periods")
  ax_loops.axvline(0.0, color="gray", lw=0.8, ls="--")
  ax_loops.set_xlabel("a")
  ax_loops.set_ylabel("count")
  ax_loops.set_title("Amount of loops versus spin")
  ax_loops.grid(True)
  ax_loops.legend(fontsize=8)

  fig_a.subplots_adjust(top=0.93, bottom=0.09, left=0.07, right=0.97)

  # lock the x-axis to the actually simulated spin range
  for ax in (ax_tau, ax_tdot, ax_prec_a, ax_loops):
      ax.set_xlim(a_vals[0], a_vals[-1])

  plt.show()
