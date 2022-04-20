import "./utils" for Counter

class Observation {
    construct new(classification, features) {
        _classification = classification
        _features = features
    }

    classification { _classification }
    features { (0..._features.count).toList }
    feature(index) { _features[index] }

    toString { "Observation(%(_classification) <- %(_features))" }
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

class Tree is Predictor {
    construct new(predictor) { _root = predictor }
    predict(observation) { _root.predict(observation) }
}

class Forest is Predictor {
    construct new() { _trees = [] }
    add(tree) { _trees.add(tree) }

    // return the consensus (most frequest) prediction of the trees
    predict(observation) {
        var counter = Counter.new()
        _trees.each { |t| counter.add(t.predict(observation)) }
        return counter.mode
    }

    toString { "Forest(trees=%(_trees.count))"}
}

class Observations {
    count { _data.count }
    features { _data[0].features }
    mode { _counter.mode }

    construct new() {
        _data = []
        _counter = Counter.new()
    }

    static from(data) {
        var instance = new()
        data.each { |o| instance.add(o) }
        return instance
    }

    add(observation) {
        _data.add(observation)
        _counter.add(observation.classification)
    }

    // a measure of how well the data has been separated by classification
    giniImpurity {
        var sum = 0.0
        for (count in _counter.counts) {
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

    // generate candidate classifiers using the given subset of features
    classifiersFor(features) {
        var classifiers = []
        for (feature in features) {
            // split the data at each observed point in the feature-space
            // todo: try something (potentially) smarter (e.g. split on median)
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

    // select the generated classifier with the best (i.e. lowest) score
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

    // select a random subset of features
    sampleFeatures() {
        var n = _data.features.count.sqrt.floor
        return _random.sample(_data.features, n)
    }

    // generate a leaf or branching stump predictor for the given data
    generatePredictor(data) {
        // if only one observation, generate a leaf returning its classification
        if (data.count <= 1) {
            return Leaf.new(data.mode)
        }

        var candidate = Candidates.new(data).bestFor(this.sampleFeatures())
        // if best candidate did not split the data, generate a leaf
        // returning the most frequent classification in the data
        if (candidate.groups.count <= 1) {
            return Leaf.new(data.mode)
        }

        var branches = {}
        for (group in candidate.groups) {
            // recursively generate predictors for each group's data subset
            branches[group.key] = this.generatePredictor(group.value)
        }
        return Stump.new(candidate.classifier, branches)
    }

    generateTree() {
        // todo: use a different (random) sample of the training data for each tree?
        return Tree.new(this.generatePredictor(_data))
    }

    generateForest(size) {
        var forest = Forest.new()
        (0...size).each { forest.add(this.generateTree()) }
        return forest
    }
}
