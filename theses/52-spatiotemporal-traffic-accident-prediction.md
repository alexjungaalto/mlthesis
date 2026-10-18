---
number: 52
author: "A. Manninen"
title: "Spatiotemporal Traffic Accident Prediction Using Deep Learning Models"
level: MSc
university: Aalto
date: "2024-06"
industry: null
aaltodoc: "https://urn.fi/URN:NBN:fi:aalto-202406234904"
video: "https://youtu.be/saahi0X6Di4"
video_views: 368
keywords: ["traffic safety", "deep learning", "neural network", "spatiotemporal prediction"]
area: spatiotemporal-forecasting
framing: supervised
confidence: high
sources: [abstract, fulltext]
review_status: draft
dictionary: [datapoint, featurevec, label, hypospace, loss, ann, baseline, overfitting, regularization, valset, testset, learnrate, gd]
---

# Spatiotemporal Traffic Accident Prediction Using Deep Learning Models

**A. Manninen** — M.Sc. thesis, Aalto University, June 2024

The City of Helsinki has set a target of halving traffic deaths and serious injuries by 2030
against a 2020 baseline, which is the premise the thesis takes as given. Meeting a target like that requires knowing *where* and *when* to intervene, which
is a forecasting problem before it is a policy one. This thesis asks whether a week's notice
of elevated accident risk, at the resolution of a city block, can be extracted from data the
city already collects.

## The learning problem

**Data points.** One [data point](https://dictionaryofml.org/terms/datapoint.html) is a
*(grid cell, week)* pair: a 1000 m × 1000 m subregion of Helsinki, paired with one week. A
road-network mask restricts prediction to cells that actually contain road, which matters
more than it sounds — without it, the model spends its capacity learning that parks and water
have no collisions.

**Features.** Each prediction uses the preceding seven days of observations for that cell,
drawn from several public sources and grouped into families the thesis ablates separately:
traffic volume, weather (temperature, rainfall), calendar (month, quarter, Finnish holidays),
land use (e.g. tree cover) and road-network structure (e.g. extent of paved road).

**Labels.** The [label](https://dictionaryofml.org/terms/label.html) is the next week's
accident count in that cell — a non-negative integer, making this regression rather than
classification. The central difficulty is that the label is *zero-inflated*: accidents are
rare, and at this spatial resolution the overwhelming majority of cell-weeks are zero. A model
that always predicts zero scores well on average error while being useless, and much of the
thesis is organised around that trap.

**Hypothesis space.** Three deep architectures are compared against a historical-average
[baseline](https://dictionaryofml.org/terms/baseline.html): a long short-term memory network (LSTM); a
convolutional neural network (CNN) paired with an LSTM, so that spatial and temporal
structure are processed by different machinery; and a transformer.

**Loss function.** Mean squared error (MSE), minimised with RMSprop at a learning
[rate](https://dictionaryofml.org/terms/learnrate.html) of 0.0001 and batch size 64, for up
to 100 epochs with early stopping after five epochs without validation improvement. Dropout
and early stopping both act as [regularisation](https://dictionaryofml.org/terms/regularization.html).
Evaluation deliberately reports MSE, mean absolute error (MAE) *and* recall together. Under
zero-inflation the error metrics and the detection metric disagree, which is the point.

## What it found

CNN-LSTM was best on MSE (0.01546) and recall (0.2764); plain LSTM was best on MAE (0.02602).
No single model dominates, and the thesis makes the sensible recommendation to pick by which
error you care about: CNN-LSTM when you want to catch accidents, LSTM when you want calibrated
magnitudes.

One honest anomaly is reported rather than smoothed over: validation
[loss](https://dictionaryofml.org/terms/loss.html) came out *lower* than training loss for
every model. The thesis works through the candidate explanations rather than leaving it as a curiosity:
more accidents fall in the training period than the validation period, and dropout is active
during training but not evaluation. That discussion is a good model of what
to do when a diagnostic looks wrong.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/saahi0X6Di4" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

[Read the thesis at Aaltodoc](https://urn.fi/URN:NBN:fi:aalto-202406234904) (open access)
