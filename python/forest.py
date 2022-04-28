import math, random, statistics
from dataclasses import dataclass
from collections import Counter, defaultdict

class CounterWithMode(Counter):
  def mode(self):
    try:
      return self.most_common(1)[0][0]
    except IndexError:
      return None

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
    self.__counter = CounterWithMode()

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
    return self.__counter.mode()

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
      # how does the given classifier group the observation?
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

  # calculate the weighted Gini Impurity (lower is better)
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

# "abstract" base class
class Predictor:
  def predict(self, observation: Observation):
    pass

# Predict the given classification, regardless of the observation
class Leaf(Predictor):
  def __init__(self, classification):
    self.__classification = classification

  def predict(self, observation):
    return self.__classification

# Branch prediction using the given classifier
class Stump(Predictor):
  def __init__(self, classifier: Classifier, branches: dict):
    self.__classifier = classifier
    self.__branches = branches

  def predict(self, observation):
    classification = self.__classifier.classify(observation)
    return self.__branches[classification].predict(observation)

# The root of a tree of predictors (e.g. stumps/branches and leafs)
class Tree(Predictor):
  def __init__(self, predictor: Predictor):
    self.__root = predictor

  def predict(self, observation):
    return self.__root.predict(observation)

# A collection of trees
class Forest(Predictor):
  def __init__(self):
    self.__trees = []

  def add(self, tree: Tree):
    self.__trees.append(tree)

  # return the consensus (most frequest) prediction of the trees
  def predict(self, observation):
    predictions = [tree.predict(observation) for tree in self.__trees]
    return statistics.mode(predictions)

class Train:
  def __init__(self, data: Observations, seed = None):
    self.__data = data
    self.__rng = random.Random(seed)

  # select a random subset of features
  def sample_features(self) -> list:
    features = self.__data.features()
    n = math.floor(math.sqrt(len(features)))
    return self.__rng.sample(features, n)

  def generate_tree(self) -> Tree:
    # todo: use a different (random) sample of the training data for each tree?
    return Tree(self.__generate_predictor(self.__data))

  def generate_forest(self, size: int) -> Forest:
    forest = Forest()
    for _ in range(0, size):
      forest.add(self.generate_tree())
    return forest

  def __generate_predictor(self, data: Observations) -> Predictor:
    # if only one observation, generate a leaf returning its classification
    if data.count() <= 1: return Leaf(data.mode())

    candidate = Candidates(data).best_for(self.sample_features())
    # if best candidate did not split the data, generate a leaf
    # returning the most frequent classification in the data
    if len(candidate.groups) <= 1: return Leaf(data.mode())

    branches = dict()
    for (classification, observations) in candidate.groups.items():
      # recursively generate predictors for each group's data subset
      branches[classification] = self.__generate_predictor(observations)
    return Stump(candidate.classifier, branches)
