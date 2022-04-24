import math
from dataclasses import dataclass
from collections import Counter, defaultdict

@dataclass(frozen=True)
class Observation:
  classification: str
  features: list[float]

  def feature(self, index):
    return self.features[index]

@dataclass(frozen=True)
class Classifier:
  feature: int
  threshold: float

  def classify(self, observation) -> bool:
    value = observation.feature(self.feature)
    return ">" if value > self.threshold else "≤"

class Observations:
  def __init__(self):
    self.__data = []
    self.__counter = Counter()

  @classmethod
  def fromList(cls, data):
    instance = cls()
    for observation in data: instance.add(observation)
    return instance

  def add(self, observation):
    self.__data.append(observation)
    self.__counter[observation.classification] += 1

  def count(self) -> int:
    return len(self.__data)

  def mode(self):
    try:
      return self.__counter.most_common(1)[0][0]
    except IndexError:
      return None

  def features(self) -> list:
    try:
      return range(0, len(self.__data[0].features))
    except IndexError:
      return []

  # a measure of how well the data has been separated by classification
  def gini_impurity(self) -> float:
    total = self.count()
    sum = 0.0
    for count in dict.values(self.__counter):
      p = count / total
      sum += (p * p)
    return (1.0 - sum)

  def __str__(self):
    return f"Observations: N={self.count()}, mode={self.mode()}, impurity={self.gini_impurity()} features={len(self.features())}"

  def partition(self, classifier) -> dict:
    groups = defaultdict(Observations)
    for observation in self.__data:
      group = classifier.classify(observation)
      groups[group].add(observation)
    return groups

  # generate candidate classifiers using the given subset of features
  def classifiers_for(self, features) -> list[Classifier]:
    classifiers = []
    for feature in features:
      # split the data at each observed point in the feature-space
      # todo: try something (potentially) smarter (e.g. split on median)
      for observation in self.__data:
        threshold = observation.feature(feature)
        classifiers.append(Classifier(feature, threshold))
    return classifiers

class Candidate:
  def __init__(self, classifier: Classifier, data: Observations):
    self.classifier = classifier
    self.groups = data.partition(classifier)

  def score(self):
    impurity = 0.0
    total = 0
    for data in self.groups.values():
      count = data.count()
      impurity += data.gini_impurity() * count
      total += count
    return impurity / total

  def __str__(self):
    groups = ', '.join([f"'{p}': {o}" for p, o in self.groups.items()])
    return f"Candidate {self.classifier}, {{{groups}}}, {self.score()}"

class Candidates:
  def __init__(self, data: Observations):
    self.__data = data

  # select the generated classifier with the best (i.e. lowest) score
  def best_for(self, features) -> Candidate:
    best = (math.inf, None)
    for classifier in self.__data.classifiers_for(features):
      candidate = Candidate(classifier, self.__data)
      score = candidate.score()
      if score < best[0]: best = (score, candidate)
    return best[1]
