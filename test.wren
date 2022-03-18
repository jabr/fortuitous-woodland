System.print("random forest in wren (https://wren.io/)")

import "os" for Process
import "io" for File

import "./forest" for Observations, Train, Candidates

var data = Observations.new()

// csv data format: feature1,feature2,...,featureN,classification
File.read(Process.arguments[0])
    .trim().split("\n")
    .map { |r| r.split(",") }
    .each { |r|
        data.add(
            r[-1], // last column is the classification
            // convert leading columns to list of numbers
            r[0..-2].map { |n| Num.fromString(n) }.toList
        )
    }

System.print("input data: %(data)")

var train = Train.new(data, 2)
var candidate = Candidates.new(data).bestFor(train.sampleFeatures())
System.print(candidate)

// todo: recursively create predictors from best candidates to form tree
// todo: create forest with trees built using different subsets of features (& observations?)
