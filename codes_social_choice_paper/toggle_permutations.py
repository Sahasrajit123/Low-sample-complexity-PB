
### Gives the set of all cases from 2^20 obtained after removing the case unique upto permutation of voter preferences and after removing cases obtained 
## after toggling bit correponding to every position as described in Lemma 20, thus case 2^20-1 and 0 are identical and only one needs to be solved.


import cvxpy as cp
import numpy as np
import math,random
import time
import itertools
from itertools import combinations, chain
 
def findsubsets(s, n):
    return list((map(frozenset, itertools.combinations(s, n))))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped



n = 6 ###no of voters used in pessimistic distortion
verbose = False
all_possibilities = []
all_perms = list(itertools.permutations(range(n)))
all_subsets = findsubsets(range(n), 3)

def subset_to_str(subset):
  if len(subset) == 0:
    return "[]"
  else:
    return str(sorted(list(subset)))

all_subset_strings = [subset_to_str(subset) for subset in all_subsets]
subset_str_to_id = {}

for i in range(len(all_subset_strings)):
  subset_str_to_id[all_subsets[i]] = i

def subset_to_id(subset):
  return subset_str_to_id[subset]


def apply_permutation(permutation, subset):
  return {permutation[x] for x in subset}

all_vals = set()
cur_iter = 0
t1 = time.time()

comp_power_set = [] ##does not contain both a set and its complement

for sets_on in powerset(all_subsets):

  new_set = tuple([x for x in all_subsets if x not in sets_on])

  if (len(new_set) > len(sets_on)): ##appending larger set
    comp_power_set.append(new_set)
  elif (len(new_set)==len(sets_on)):
    if (new_set not in comp_power_set and sets_on not in comp_power_set): ##not contain both itself and complement
      comp_power_set.append(sets_on)

print ("Len after redn",len(comp_power_set))


for sets_on in comp_power_set:


  cur_iter += 1
  if (cur_iter % 1000) == 0:
    t2 = time.time()
    print(cur_iter, " time ", t2 - t1, end=" ", flush = True)
    t1 = t2
    if cur_iter % 50000 == 0:
      print()
  min_val = 1 << len(all_subsets)
  min_perm = 0
  for perm in all_perms:
    val = 0
    new_sets_on = [frozenset(apply_permutation(perm, subset)) for subset in sets_on]
    for subset in new_sets_on:
      val += (1 <<  subset_to_id(subset))
    if val < min_val:
      min_val = val
      min_perm = perm
  if (min_val in all_vals):
    if verbose:
      print("skipping ", sets_on, "since it gives value ", min_val,  " with permutation ", min_perm)
  else:
    if verbose:
      print("adding ", sets_on, " with value", min_val )
  all_vals.add(min_val)

print()
if verbose:
  print(all_vals)
print(len(all_vals))

