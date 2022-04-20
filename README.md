# fortuitous-woodland

A simple [random forest](https://en.wikipedia.org/wiki/Random_forest) classifier in multiple languages.

* [Wren](https://wren.io/):

  ### Testing model generation

  ```sh
  wren_cli wren/test.wren example-data.csv
  ```

  ### Import module in code

  ```wren
  import "./forest" for Observations, Observation, Classifier
  // @todo
  ```

## References

* Example data
  - [Connectionist Bench (Sonar, Mines vs. Rocks) Data Set](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
* [How to Implement Random Forest From Scratch in Python](https://machinelearningmastery.com/implement-random-forest-scratch-python/)

## License

This project is licensed under the terms of the [MIT license](LICENSE.txt).
