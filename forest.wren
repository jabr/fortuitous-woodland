class Observation {
    construct new(classification, features) {
        _classification = classification
        _features = features
    }

    classification { _classification }
    features { (0..._features.count).toList }
    feature(index) { _features[index] }
}

class Classifier {
    construct new(feature, threshold) {
        _feature = feature
        _threshold = threshold
    }

    classify(observation) {
        var value = observation.feature(_feature)
        return value > _threshold ? ">" : "≤"
    }

    toString { "Classifier(*%(_feature) > %(_threshold))" }
}

// "abstract" base class
class Predictor {
    predict(observation) {}
}

class Leaf is Predictor {
    construct new(classification) { _classification = classification }
    predict(observation) { _classification }
}

class Stump is Predictor {
    construct new(classifier, branches) {
        _classifier = classifier
        _branches = branches
    }

    predict(observation) {
        return _branches[_classifier.classify(observation)].predict(observation)
    }
}

class Observations {
    count { _data.count }
    features { _data[0].features }

    construct new() {
        _data = []
        _counts = {}
    }

    add(observation) {
        _data.add(observation)
        var c = observation.classification
        _counts[c] = (_counts[c] || 0) + 1
    }

    add(classification, features) {
        this.add(Observation.new(classification, features))
    }

    mode {
        var mode = "∅"
        var max = 0
        for (entry in _counts) {
            if (entry.value > max) {
                max = entry.value
                mode = entry.key
            }
        }
        return mode
    }

    giniImpurity {
        var sum = 0.0
        for (count in _counts.values) {
            var p = count / this.count
            sum = sum + (p * p)
        }
        return (1.0 - sum)
    }

    toString { "Observations(N=%(this.count), mode=%(this.mode), impurity=%(this.giniImpurity), features=%(this.features.count))" }

    partition(classifier) {
        var groups = {}
        for (observation in _data) {
            // how does the given classifier group the observation?
            var group = classifier.classify(observation)
            (groups[group] = groups[group] || Observations.new()).add(observation)
        }
        return groups
    }

    classifiersFor(features) {
        var classifiers = []
        for (feature in features) {
            for (observation in _data) {
                var threshold = observation.feature(feature)
                classifiers.add(Classifier.new(feature, threshold))
            }
        }
        return classifiers
    }
}

class Candidate {
    classifier { _classifier }
    groups { _groups }

    construct new(classifier, data) {
        _classifier = classifier
        _groups = data.partition(_classifier)
    }

    // calculate the weighted Gini Impurity (lower is better)
    score {
        var impurity = 0.0
        var total = 0
        for (data in _groups.values) {
            impurity = impurity + data.giniImpurity * data.count
            total = total + data.count
        }
        return (impurity / total)
    }

    toString { "Candidate(%(this.classifier), %(this.groups), score=%(this.score))"}
}

class Candidates {
    construct new(data) {
        _data = data
    }

    bestFor(features) {
        var best = [ Num.infinity, Null ]
        for (classifier in _data.classifiersFor(features)) {
            var candidate = Candidate.new(classifier, _data)
            var score = candidate.score
            if (score < best[0]) { best = [ score, candidate ] }
        }
        return best[1]
    }
}

import "random" for Random

class Train {
    construct new(data) {
        _data = data
        _random = Random.new()
    }

    construct new(data, seed) {
        _data = data
        _random = Random.new(seed)
    }

    sampleFeatures() {
        var n = _data.features.count.sqrt.floor
        return _random.sample(_data.features, n)
    }
}
