System.print("random forest in wren (https://wren.io/)")

import "os" for Process
import "io" for File

import "./forest" for Observations, Observation, Classifier, Candidate

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
