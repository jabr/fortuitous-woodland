class Counter {
    construct new() { _counts = {} }

    counts { _counts.values }
    mode { _counts.reduce { |m, e| e.value > m.value ? e : m }.key } // the most frequent value

    add(value) {
        _counts[value] = (_counts[value] || 0) + 1
    }
}

class Statistics {
    construct on(list) { _list = list }

    sum { _list.reduce { |a,b| a + b } }
    mean { this.sum / _list.count }

    // split the data into k number of groups
    folds(k) {
        var n = (_list.count / k).floor
        return (0...k).map { |fold|
            var s = fold * n
            return _list[fold == k-1 ? s..-1 : s...s+n]
        }.toList
    }

    // generate k-fold splits, returning pairs of [fold,other] data subsets
    splits(k) {
        var folds = this.folds(k)
        return (0...k).map { |i|
            // combine all other data...
            var other = []
            for (j in 0...k) {
                // excluding the current fold of data
                if (i != j) other.addAll(folds[j])
            }
            return [ folds[i], other ]
        }.toList
    }
}

class CSV {
    static read(path) {
        import "io" for File
        return File.read(path).trim().split("\n").map { |r| r.split(",") }
    }
}
