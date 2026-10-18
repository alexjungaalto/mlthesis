---
number: 126
author: "T. Kokkonen"
title: "Classifying Restaurant Menu Items with Supervised Learning"
level: MSc
university: Aalto
date: "2021-01"
industry: null
aaltodoc: "https://aaltodoc.aalto.fi/handle/123456789/102433"
video: "https://youtu.be/TaYlv6MFoeU"
video_views: 445
keywords: ["supervised learning", "machine learning", "classification", "restaurant", "food"]
area: nlp-text-mining
framing: supervised
confidence: high
sources: [abstract, fulltext]
review_status: draft
dictionary: [supervisedlearning, datapoint, feature, label, hypospace, svm, randomforest, logreg, ann, loss, kfoldcv, overfitting, samplesize]
---

# Classifying Restaurant Menu Items with Supervised Learning

**T. Kokkonen** — M.Sc. thesis, Aalto University, January 2021

Part of [Natural language processing and text mining](../areas/nlp-text-mining.md).

Food delivery platforms carry hundreds of thousands of dishes written as free text by
restaurants, for humans. Nothing in "Tonnikalasalaatti" tells a search index that it is a
salad, or fish, or neither-if-you-are-vegetarian. Without machine-readable tags, the
platform-wide operations a user expects cannot be built: filtering, dietary search, category
browsing. This thesis asks how far ordinary
[supervised learning](https://dictionaryofml.org/terms/supervisedlearning.html) gets you on
that tagging problem, using real menu data from Wolt.

## The learning problem

**Data points.** One [data point](https://dictionaryofml.org/terms/datapoint.html) is a
single menu item from a Finnish restaurant on the Wolt platform, carrying four short text
fields: its category name, dish name, and description among them.

**Features.** The four text fields are concatenated into a single bag-of-words vector. A
second [feature](https://dictionaryofml.org/terms/feature.html) representation was tried as a
comparison: averaged pre-trained Word2Vec embeddings from the Google News corpus, on the
theory that embeddings would generalise better. They did not: 50.5% subset accuracy against
63.2% for bag-of-words. An ablation over the four fields showed all of them contribute;
category information alone is not enough to label a dish.

**Labels.** The dataset arrived unlabelled, so the labelling scheme is itself part of the
thesis. Twenty [labels](https://dictionaryofml.org/terms/label.html) were defined and applied
by hand: *sushi, salad, Thai, Indian, sandwich, Vietnamese, fish, pasta, dessert, smoothie,
ramen, bakery, soup, pizza, burger, meat, alcohol, drink, grocery, food*. Each was chosen so
it has enough positive examples, is unambiguous, and is useful for filtering. Each item may
carry several labels at once, making this **multi-label** rather than multi-class.

**Hypothesis space.** The multi-label problem is reduced to twenty independent binary problems
via Binary Relevance, and six classifier families are compared within it:
[support vector machines](https://dictionaryofml.org/terms/svm.html) with linear, radial basis
function (RBF) and sigmoid kernels, multinomial Naive Bayes,
[random forest](https://dictionaryofml.org/terms/randomforest.html), linear and
[logistic regression](https://dictionaryofml.org/terms/logreg.html), and a
multi-layer perceptron (MLP), a feed-forward
[artificial neural network](https://dictionaryofml.org/terms/ann.html) with a single 100-unit
hidden layer and 20 output units.

**Loss function.** Each family brings its own training objective; the thesis is explicit that
the MLP minimises a mean-squared-error cost. Comparison across families therefore happens at
the level of evaluation metrics, not [loss](https://dictionaryofml.org/terms/loss.html) —
subset accuracy, Hamming loss, and micro- and macro-averaged F1, all under five-fold
[cross-validation](https://dictionaryofml.org/terms/kfoldcv.html).

## What it found

The linear-kernel SVM won on every metric but one: 63.2% subset accuracy, 0.890 F-measure,
93.8% precision, 84.8% recall. The MLP came second (58.7%, 0.867). The RBF kernel reached only 51.8% subset accuracy
and 0.835 F-measure, well below the linear kernel. The extra capacity of a non-linear kernel
costs accuracy when the feature space is already high-dimensional and sparse.

Two secondary findings are worth more than the leaderboard. Almost every classifier was badly
[overfitting](https://dictionaryofml.org/terms/overfitting.html), reaching ~99% accuracy on the
training set; the two that did not, logistic regression and Naive Bayes, scored below every other
classifier on subset accuracy.
And the learning curve flattens at roughly 750 labelled examples. Going from 750 to 1,000
changed the F-measure by 0.002. For anyone planning a labelling budget, that
second result is the one to take away.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/TaYlv6MFoeU" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

[Read the thesis at Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/102433) (open access)
