print("random forest in Python")

from forest import Observation, Observations

import sys, csv
rows = []
for row in csv.reader(open(sys.argv[1])):
  *features, classification = row
  features = [float(f) for f in features]
  rows.append(Observation(classification, features))

# @todo

# print(rows[-1])

print(Observations())

data = Observations.fromList(rows)
print(data)

import forest
candidates = forest.Candidates(data)
candidate = candidates.best_for([0])
print(candidate)
