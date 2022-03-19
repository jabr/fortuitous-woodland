System.print("random forest in wren (https://wren.io/)")

import "os" for Process
import "io" for File

import "./forest" for Observations, Observation, Train

// csv data format: feature1,feature2,...,featureN,classification
var rows = File.read(Process.arguments[0])
    .trim().split("\n").map { |r| r.split(",") }.map { |r|
        return Observation.new(
            // last column is the classification
            r[-1],
            // convert leading columns to list of numbers
            r[0..-2].map { |n| Num.fromString(n) }.toList
        )
    }

var data = Observations.from(rows)
System.print("input data: %(data)")

var train = Train.new(data, 2)
var forest = train.generateForest(5)
System.print(forest)

var o1 = rows.toList[-1]
System.print("prediction %(forest.predict(o1)) :: %(o1)")
