"""Analysis of head-score tables: heatmaps, checkpoint sweep curves, correlations, direction angles, report.

Everything here is torch-free and works on the JSON / npz artifacts written by the probe, probe-stats and
directions commands, so it runs on a laptop without the models or the feature shards.
"""
