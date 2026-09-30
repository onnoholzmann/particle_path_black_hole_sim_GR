from calc_path import *
import numpy as np
import time
import scipy

time_start = time.time()

# init objects
w, h = 1280, 720

attractor_mass = 1
attractors = [Attractor(attractor_mass, (w/2, h/2), a/100, convert_a=True) for a in range(-99, 100)]
# orbiters = [Relative_object(attractor, (200, 0, np.pi/2), (-0.000002, 0.0002, 0), False) for attractor in attractors]
orbiters = [Relative_object(attractor, (200, 0, np.pi/2), (0, 0, 0), True) for attractor in attractors]
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


# gen data
for orbiter in orbiters:
  current_cache = []
  for _ in range(1000):
    # orbiter.update(100, 0.05)
    # current_cache.append([orbiter.pos.copy(), orbiter.speed.copy()])
    orbiter.update(100, 0.05, True)
    current_cache.append([orbiter.interp_pos.copy(), orbiter.interp_speed.copy()])
  orbiters_cache.append(current_cache)
  # print("finished orbiter with a =", orbiter.attractor.a)

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
  if len(deltas) > 0:
    return deltas[i][j-i-1][snapshot_index], d, err
  else:
    return [orbiters_cache[i][snapshot_index][0] - orbiters_cache[j][snapshot_index][0], orbiters_cache[i][snapshot_index][1] - orbiters_cache[j][snapshot_index][1]], d, err

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
