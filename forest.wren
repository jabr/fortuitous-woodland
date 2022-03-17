System.print("random forest in wren (https://wren.io/)")

import "os" for Process
import "io" for File
import "random" for Random
var random = Random.new()

class Observation {
    construct new(classification, features) {
        _classification = classification
        _features = features
    }

    classification { _classification }
    feature(index) { _features[index ] }
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
}

// interface/abstract class
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

    construct new() {
        _data = []
        _counts = {}
    }

    add(observation) {
        _data.add(observation)
        var c = observation.classification
        _counts[c] = (_counts[c] || 0) + 1
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

    toString { "Observations(N=%(this.count), mode=%(this.mode), impurity=%(this.giniImpurity))" }

    partition(classifier) {
        var groups = {}
        for (observation in _data) {
            // how does the given classifier group the observation?
            var group = classifier.classify(observation)
            (groups[group] = groups[group] || Observations.new()).add(observation)
        }
        return groups
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
}

var data = Observations.new()

// csv data format: feature1,feature2,...,featureN,classification
File.read(Process.arguments[0])
    .trim().split("\n")
    .map { |r| r.split(",") }
    .map { |r| Observation.new(
        r[-1], // last column is the classification
        // convert leading columns to list of numbers
        r[0..-2].map { |n| Num.fromString(n) }.toList
    ) }
    .each { |o| data.add(o) }

System.print("input data: %(data)")

var candidate = Candidate.new(Classifier.new(1, 0.01), data)
System.print(candidate.groups)
System.print(candidate.score)

// todo: generate candidate classifiers and select best score
// todo: recursively create predictors from best candidates to form tree
// todo: create forest with trees built using different subsets of features (& observations?)
