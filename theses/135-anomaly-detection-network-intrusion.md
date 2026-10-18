---
number: 135
author: "M. Mishin"
title: "Anomaly Detection Algorithms and Techniques for Network Intrusion Detection Systems"
level: MSc
university: Aalto
date: "2020-08"
industry: "Ericsson"
aaltodoc: "https://aaltodoc.aalto.fi/handle/123456789/46076"
video: "https://youtu.be/Z_lBOjkaSPk"
video_views: 426
keywords: ["anomaly detection", "network intrusion detection", "neural networks", "autoencoders", "network-flow aggregation", "semi-supervised learning"]
area: anomaly-fraud-detection
framing: anomaly-detection
confidence: high
sources: [abstract, fulltext, video]
review_status: draft
dictionary: [datapoint, featurevec, label, hypospace, loss, autoencoder, gmm, pca, ensemble, testset, hyperparameter, unsupervisedlearning]
---

# Anomaly Detection Algorithms and Techniques for Network Intrusion Detection Systems

**M. Mishin** — M.Sc. thesis, Aalto University, August 2020 (with Ericsson)

Part of [Anomaly, intrusion and fraud detection](../areas/anomaly-fraud-detection.md).

An intrusion detection system has to flag attacks it has never seen, which rules out training
a classifier on a catalogue of known ones. The standard move is to learn what normal traffic
looks like and score everything by how poorly it fits. This thesis tests how well that works
in practice, comparing deep autoencoding models against classical methods on three modern
intrusion datasets — and pays unusual attention to how the benchmark itself is built.

## The learning problem

This is not standard supervised learning, so the decomposition needs one adjustment: there is
no label at training time. The models are trained *semi-supervised*, on traffic that is mostly
normal, and labels appear only for threshold selection and evaluation.

**Data points.** Either a single network flow, or a *group* of flows sharing a source internet
protocol (IP) address within a sliding time window. Which of the two is the
[data point](https://dictionaryofml.org/terms/datapoint.html) is one of the thesis's two
experimental axes. Three public sources are used: CTU-13, CICIDS 2017 and IoT-23, the last
capped at 5 million points for tractability.

**Features.** In the unaggregated case, the numeric flow attributes, with IP address and port
discarded as having no straightforward encoding. In the aggregated case, descriptive
statistics computed per group over the window — plus new features that only exist at group
level, such as the number of unique destination IPs and their entropy. Those two are exactly
the signals a port scan produces and a single flow cannot show.

**Labels.** Binary, *attack* or *benign*, used only for setting the detection threshold and
for evaluation. For aggregated data the [label](https://dictionaryofml.org/terms/label.html)
is inherited by a deliberately conservative rule: a group is *attack* if **any** flow in it
was.

**Hypothesis space.** Three deep models against two classical baselines. The deep models are
a basic [autoencoder](https://dictionaryofml.org/terms/autoencoder.html) (AE), a variational
autoencoder (VAE), and AEGMM, which couples an autoencoder with a
[Gaussian mixture](https://dictionaryofml.org/terms/gmm.html). The baselines are isolation
forest and a Mahalanobis-distance method applied after
[PCA](https://dictionaryofml.org/terms/pca.html).

**Loss function.** Reconstruction error for the plain autoencoder; the ELBO for the VAE,
combining a reconstruction term with a KL term pulling the latent distribution towards a
standard normal prior. The [loss](https://dictionaryofml.org/terms/loss.html) serves two
roles here, which is characteristic of the approach: the same quantity minimised during
training becomes the anomaly score at test time. Isolation forest instead scores by expected
path length — anomalies are easier to isolate, so they sit closer to the root.

## What it found

The deep autoencoding models, AE and VAE in particular, scored higher than the classical
methods across all three datasets, and aggregating flows before scoring improved every
reported metric. Both are
useful, transferable findings.

The methodology chapter is the more transferable part. It proposes a repeatable recipe for
constructing intrusion-detection benchmarks: how to define an anomaly, and how to split train
from test. It argues for splitting *by scenario* rather than at random, so models are
evaluated on attack types absent from their training data and the original contamination rate
is preserved. That is a harder and more honest test than the usual random split, and the
results under it are correspondingly more sober. The main caveat concerns deployment: false alarm
rates remain high enough that a second filtering stage is needed before the output is usable.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/Z_lBOjkaSPk" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

[Read the thesis at Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/46076) (open access)
