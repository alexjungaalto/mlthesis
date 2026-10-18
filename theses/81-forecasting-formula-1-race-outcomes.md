---
number: 81
author: "L. García Tejada"
title: "Applying Machine Learning to Forecast Formula 1 Race Outcomes"
level: MSc
university: Aalto
date: "2023-08"
industry: null
aaltodoc: "https://aaltodoc.aalto.fi/handle/123456789/122937"
video: "https://youtu.be/q01GzbSDsSg"
video_views: 574
keywords: ["machine learning", "data science", "artificial neural networks", "random forest", "predictive model", "Formula 1"]
area: time-series-forecasting
framing: supervised
confidence: high
sources: [abstract, fulltext, video]
review_status: draft
dictionary: [binclass, datapoint, featurevec, label, hypospace, logloss, randomforest, svm, ann, kfoldcv, overfitting, cm]
---

# Applying Machine Learning to Forecast Formula 1 Race Outcomes

**L. García Tejada** — M.Sc. thesis, Aalto University, August 2023

Part of [Time series and forecasting](../areas/time-series-forecasting.md).

A pit stop is one of the few decisions in a Formula 1 race that a team fully controls, and
getting the lap wrong can cost places that no amount of pace recovers. The question this
thesis asks is whether the decision can be anticipated from race telemetry and standings
alone — not to replace the strategist, but to see how much of the call is recoverable from
data that is public while the race is still running.

## The learning problem

**Data points.** One [data point](https://dictionaryofml.org/terms/datapoint.html) is a
single *driver-lap*: one driver, on one lap, of one race, drawn from the 2019–2022 seasons.
Laps run in wet conditions were removed, as were drivers who retired or were lapped more
than three times, on the grounds that their pit-stop behaviour is driven by something other
than strategy.

**Features.** Twenty-one variables per driver-lap, mixing categorical, numerical, binary and
time-based information about the race, the driver and the lap itself. Categorical variables
were one-hot encoded, numerical ones standardised; missing qualifying positions were filled
with the starting position.

**Labels.** Two separate binary [labels](https://dictionaryofml.org/terms/label.html), each
giving its own [binary classification](https://dictionaryofml.org/terms/binclass.html) task.
*Has pit stop* records whether the driver pitted on that lap. *Good pit stop* records whether
the driver's position improved or held after pitting. Both are heavily imbalanced: only about 3%
of laps contain a pit stop, and roughly half of those count as good. That imbalance shapes
every modelling choice that follows.

**Hypothesis space.** Three families were compared rather than one chosen in advance: a
[random forest](https://dictionaryofml.org/terms/randomforest.html), a
[support vector machine](https://dictionaryofml.org/terms/svm.html) with several kernels, and
a feed-forward [artificial neural network](https://dictionaryofml.org/terms/ann.html) with
ReLU hidden layers and a sigmoid output.

**Loss function.** The neural network was trained with binary cross-entropy, the
[logistic loss](https://dictionaryofml.org/terms/logloss.html), and stopped early when
validation loss failed to improve for five consecutive epochs. Because accuracy is close to
meaningless at a 3% positive rate, model selection was driven by the F1 score throughout,
with class weighting used to stop the majority class from dominating training. Hyperparameters
were chosen by grid search under five-fold
[cross-validation](https://dictionaryofml.org/terms/kfoldcv.html), and the three families were
compared on a held-out test set not used for tuning.

## What it found

The random forest reached AUC values of 0.95 and 0.96 on the two labels, but the thesis is
careful to point out that this is largely an artefact of the imbalance. On the F1 score, which unlike accuracy is not
inflated by the majority class, recall was the weak point: the models miss a substantial share of
actual pit stops, and performance on *good pit stop* was worse than on *has pit stop*.

The conclusion is appropriately modest, and is the interesting part: these models work as
decision-support, flagging laps worth considering, rather than as standalone predictors. The
residual uncertainty in a race is not a modelling deficiency to be engineered away.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/q01GzbSDsSg" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

[Read the thesis at Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/122937) (open access)
