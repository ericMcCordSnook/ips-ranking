# ips-ranking
Implementation of Mallow's Weighted Kendall Model for computing distances between ranking and finding most likely ground truths for simulated and real datasets.

Nikhil Bhaip & Eric McCord-Snook
MallowsWeightedKendall.py

This program is used to calculate probabilities of rankings with different weights.
Weights are abstract classes that contain information on weighting structure based on two parameters.


A Ranking is a class that holds information on a particular ranking and its properties. All rankings will have a
Weight class to calculate operations related to weights like getting the sum of weights.

A MultiRanking is a class that holds multiple Ranking objects. We can use it calculate the log-likelihood.
