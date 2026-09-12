from calc_path import *
import numpy as np
import time
import scipy

time_start = time.time()

# init objects
w, h = 1280, 720
attractors = [Attractor(1, (w/2, h/2), a/100, convert_a=True) for a in range(-99, 100)]
orbiters = [Relative_object(attractor, (200, 0, np.pi/2), (-0.000002, 0.0002, 0), False) for attractor in attractors]
orbiters_cache = []

# gen data
for orbiter in orbiters:
  current_cache = []
  for _ in range(1000):
    orbiter.update(100, 0.05)
    current_cache.append([orbiter.pos.copy(), orbiter.speed.copy()])
  orbiters_cache.append(current_cache)
  print("finished orbiter with a =", orbiter.attractor.a)

print(f'finished orbiter calc in {time.time()-time_start} seconds')
time_start = time.time()

# calc delta's
deltas = []
for i in range(len(orbiters_cache)):
  orbiter_deltas = []
  for j in range(i+1, len(orbiters_cache)):
    orbiter_deltas.append([[orbiters_cache[i][x][y] - orbiters_cache[j][x][y] for y in range(2)] for x in range(len(orbiters_cache[j]))])
  deltas.append(orbiter_deltas)

print("deltas_finished")
print(f'finished delta calc in {time.time()-time_start} seconds')

def lookup_deltas(i, j, snapshot_index):
  if i < 0:
    i = len(deltas) + i
  if j < 0:
      j = len(deltas) + j
  # returns pos, speed
  assert i != j
  if i > j:
    i, j = j, i
  return deltas[i][j-i-1][snapshot_index]

print(lookup_deltas(0, 99, 999))
print(lookup_deltas(99, -1, 999))
print(lookup_deltas(0, -1, 999), '\n')

print(lookup_deltas(0, 99, 0))
print(lookup_deltas(99, -1, 0))
print(lookup_deltas(0, -1, 0), '\n')

print(lookup_deltas(0, 99, 1))
print(lookup_deltas(99, -1, 1))
print(lookup_deltas(0, -1, 1))
