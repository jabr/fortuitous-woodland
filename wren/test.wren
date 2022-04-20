System.print("random forest in wren (https://wren.io/)")

import "os" for Process
import "random" for Random

import "./forest" for Observation, Observations, Train
import "./utils" for CSV, Statistics

class Evaluate {
    forest { _forest }
    construct new(trainingRows, size, seed) {
        var data = Observations.from(trainingRows)
        // System.print("input data: %(data)")
        var train = Train.new(data, seed)
        _forest = train.generateForest(size)
    }

    accuracy(testingRows) {
        var correct = 0
        for (observation in testingRows) {
            var prediction = _forest.predict(observation)
            // System.print("= %(prediction) :: %(observation)")
            if (prediction == observation.classification) correct = correct + 1
        }
        return correct / testingRows.count
    }
}

// csv data format: feature1,feature2,...,featureN,classification
var rows = CSV.read(Process.arguments[0]).map { |r|
        return Observation.new(
            // last column is the classification
            r[-1],
            // convert leading columns (the features) to list of numbers
            r[0..-2].map { |n| Num.fromString(n) }.toList
        )
    }.toList

var seed = 2
var kFold = 5
var rng = Random.new(seed)

for (size in [1, 5, 10]) {
    System.print("Trees: %(size)")
    rng.shuffle(rows)
    var scores = []
    for (split in Statistics.on(rows).splits(kFold)) {
        var testingRows = split[0]
        var trainingRows = split[1]
        var eval = Evaluate.new(trainingRows, size, seed)
        // System.print(eval.forest)

        scores.add(eval.accuracy(testingRows))
    }

    System.print("Scores: %(scores)")
    System.print("Mean: %(Statistics.on(scores).mean)")
}
