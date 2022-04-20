from dataclasses import dataclass
from collections import Counter

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
  __data = []
  __counter = Counter()

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

  def gini_impurity(self) -> float:
    total = float(self.count())
    sum = 0.0
    for count in dict.values(self.__counter):
      p = count / total
      sum += (p * p)
    return (1.0 - sum)

  def __str__(self):
    return f"Observations: N={self.count()}, mode={self.mode()}, impurity={self.gini_impurity()} features={len(self.features())}"
