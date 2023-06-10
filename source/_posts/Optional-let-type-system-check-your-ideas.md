---
title: 'Optional: let type system check your ideas'
date: 2023-06-10 10:45:38
tags: Optional<>
---

This blog talks about one advantage of `Optional<>`. It's nothing fancy and just warps up pieces of code into a new class. Here comes the question: why should I use it? I can do all on my own! The following two example will demonstrate how to use `Optional<>` to express the idea of empty value and to prevent logical flaws.

----------

Let's start with simpler case: the binary search. This algorithm utilizes the order of the given list and searches the index of the target. If not found, -1 will be return. So far, so good. As -1 isn't a valide value of indices.

```java
public static int bSearch(List<Integer> list, int target) {
        int ret = -1;

        int low = 0;
        int high = list.size();
        int mid = 0;

        while (low <= high) {
            mid = (low + high) / 2;
            if (list.get(mid) == target) {
                return mid;
            } else if (list.get(mid) < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return ret;
    }
```

We can refactor the code with `Optional<>`. The logic here is simple. We need a variable to store the result, but it could be empty. The `Optional<Integer>` emphasised the target coulde not be in the list instead of putting a misleading -1.

```java
public static Optional<Integer> search(List<Integer> list, int target) {
        Optional<Integer> ret = Optional.empty();

        int low = 0;
        int high = list.size();
        int mid = 0;

        while (low <= high) {
            mid = (low + high) / 2;
            if (list.get(mid) == target) {
                return Optional.of(mid);
            } else if (list.get(mid) < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return ret;
    }
```

In short, this example shows how to take the advantage of `Optional<>` to express the uncertain or un-computed idea in coding.

However, this example cannot even convince me to use `Optional<>` anywhere. An agreement on -1 does work. Why should I use another box to do that?

Here comes the other example.

It's a Java version solution for the `set-covering` problem on page 151 [1]. It tends to find the approximately letest stations covering all the states using the greedy algorithm. Everytime it goes through the map `stations` to find one that can cover most states in `statesNeeded`. Once founded, covered states will be removed from `statesNeeded`, and the station name will be put in `finalStations`. Repeat, until `statesNeeded` is empty.

Now, look at this code and tell whether it is robust.

```java
public static void main(String... args) {
    var statesNeeded = new HashSet<>(Arrays.asList("mt", "wa", "or", "id", "nv", "ut", "ca", "az"));
    var stations = new LinkedHashMap<String, Set<String>>();

    stations.put("kone", new HashSet<>(Arrays.asList("id", "nv", "ut")));
    stations.put("ktwo", new HashSet<>(Arrays.asList("wa", "id", "mt")));
    stations.put("kthree", new HashSet<>(Arrays.asList("or", "nv", "ca")));
    stations.put("kfour", new HashSet<>(Arrays.asList("nv", "ut")));
    stations.put("kfive", new HashSet<>(Arrays.asList("ca", "az")));

    var finalStations = new HashSet<String>();
    while (!statesNeeded.isEmpty()) {
        String bestStation = null;
        var statesCovered = new HashSet<String>();

        for (var station : stations.entrySet()) {
            var covered = new HashSet<>(statesNeeded);
            covered.retainAll(station.getValue());

            if (covered.size() > statesCovered.size()) {
                bestStation = station.getKey();
                statesCovered = covered;
            }
        }
        statesNeeded.removeIf(statesCovered::contains);
        finalStations.add(bestStation);
    }
    System.out.println(finalStations); // [ktwo, kone, kthree, kfive]
}

```

Of course not. It just works fine with this input. The weakness is `String bestStation = null;`. In some scenarios, `finalStations.add(bestStation);` adds a null. If initialize `bestStation` with empty String "", the case is the same, and it just replaces the null to "". However, the set `statesNeeded` works as your intention. If `statesCovered` is empty, saying no element is going to be remove from `statesNeeded`, then `statesNeeded` stays the same. Because empty set is an [identity element](https://en.wikipedia.org/wiki/Identity_element) of set operations.

In most of the case, we need to filter out empty strings. [Like this](https://github.com/egonSchiele/grokking_algorithms/blob/master/08_greedy_algorithms/java/01_set_covering/src/SetCovering.java). Different from the binary search case, this time you may forget the filter operation.

```java
public static void main(String... args) {
    var statesNeeded = new HashSet<>(Arrays.asList("mt", "wa", "or", "id", "nv", "ut", "ca", "az"));
    var stations = new LinkedHashMap<String, Set<String>>();

    stations.put("kone", new HashSet<>(Arrays.asList("id", "nv", "ut")));
    stations.put("ktwo", new HashSet<>(Arrays.asList("wa", "id", "mt")));
    stations.put("kthree", new HashSet<>(Arrays.asList("or", "nv", "ca")));
    stations.put("kfour", new HashSet<>(Arrays.asList("nv", "ut")));
    stations.put("kfive", new HashSet<>(Arrays.asList("ca", "az")));

    var finalStations = new HashSet<String>();
    while (!statesNeeded.isEmpty()) {
        String bestStation = null;
        var statesCovered = new HashSet<String>();

        for (var station : stations.entrySet()) {
            var covered = new HashSet<>(statesNeeded);
            covered.retainAll(station.getValue());

            if (covered.size() > statesCovered.size()) {
                bestStation = station.getKey();
                statesCovered = covered;
            }
        }
        statesNeeded.removeIf(statesCovered::contains);

        if (bestStation != null) {
            finalStations.add(bestStation);
        }
    }
    System.out.println(finalStations); // [ktwo, kone, kthree, kfive]
}
```

If you using the `Optional<>`, the compiler/analyzer can warn you.

```java
public static void main(String... args) {
    var statesNeeded = new HashSet<>(Arrays.asList("mt", "wa", "or", "id", "nv", "ut", "ca", "az"));
    var stations = new LinkedHashMap<String, Set<String>>();

    stations.put("kone", new HashSet<>(Arrays.asList("id", "nv", "ut")));
    stations.put("ktwo", new HashSet<>(Arrays.asList("wa", "id", "mt")));
    stations.put("kthree", new HashSet<>(Arrays.asList("or", "nv", "ca")));
    stations.put("kfour", new HashSet<>(Arrays.asList("nv", "ut")));
    stations.put("kfive", new HashSet<>(Arrays.asList("ca", "az")));

    var finalStations = new HashSet<String>();
    while (!statesNeeded.isEmpty()) {
        Optional<String> bestStation = Optional.empty();
        Set<String> statesCovered = Collections.emptySet();

        for (var station : stations.entrySet()) {
            HashSet<String> covered = new HashSet<>(statesNeeded);
            covered.retainAll(station.getValue());

            if (covered.size() > statesCovered.size()) {
                bestStation = Optional.of(station.getKey());
                statesCovered = covered;
            }
        }

        statesNeeded.removeIf(statesCovered::contains);

        bestStation.ifPresent(finalStations::add);

    }
    System.out.println(finalStations); // [ktwo, kone, kthree, kfive]
}
```

----------

# Reference
[1] A. Y. Bhargava, Grokking algorithms: an illustrated guide for programmers and other curious people. Shelter Island: Manning, 2016.
