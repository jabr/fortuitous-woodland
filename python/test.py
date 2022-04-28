print("random forest in Python")

from forest import Observation, Observations
import sys, csv, random, statistics

seed = 2
k_fold = 5
random.seed(seed)

rows = []
for row in csv.reader(open(sys.argv[1])):
  *features, classification = row
  features = [float(f) for f in features]
  rows.append(Observation(classification, features))

# generate k-fold splits, returning pairs of [fold,other] data subsets
def k_fold_splits(data: list, k: int) -> list[(list,list)]:
  splits = []
  o = 0
  n, r = divmod(len(data), k)
  for i in range(0,k):
    s = n + int(i < r)
    e = o + s
    splits.append((data[o:e], data[:o] + data[e:]))
    o = e
  return splits

def evaluate(size: int, seed, split: list):
  testing, training = split
  # todo: train
  correct = 0
  for observation in testing:
    if random.choice((True, False)): correct += 1
  return correct / len(testing)

for size in [1, 5, 10]:
  print(f'Trees: {size}')
  random.shuffle(rows)
  splits = k_fold_splits(rows, k_fold)
  scores = [evaluate(size, seed, split) for split in splits]

  print(f'Scores: {scores}')
  print(f'Mean: {statistics.mean(scores)}')
