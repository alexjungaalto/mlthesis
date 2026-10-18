---
number: 133
author: "J. Laiho"
title: "Recognizing Thoughts from Bioelectric Patterns? A Brain-Computer Interface with Deep Learning"
level: MSc
university: Aalto
date: "2020-08"
industry: "Accenture Liquid Studio (NL)"
aaltodoc: "https://aaltodoc.aalto.fi/handle/123456789/46105"
video: "https://youtu.be/IouXWsJsm0Q"
video_views: 788
keywords: ["brain-computer interfaces", "convolutional neural networks", "motor imagery", "deep learning", "transfer learning", "electroencephalography"]
area: health-biosignals
framing: supervised
confidence: medium
sources: [abstract, video]
review_status: draft
review_notes: >-
  Full text is a scanned PDF with no text layer, so the training loss could not be
  confirmed from the thesis itself and is left unstated below. Categorical cross-entropy
  is the overwhelmingly likely choice for 4-class softmax output, but it is NOT asserted
  here — confirm against the thesis before publishing.
dictionary: [datapoint, featurevec, label, hypospace, classification, ann, featlearn, transferlearning, pretraining, accuracy, generalization, distshift]
---

# Recognizing Thoughts from Bioelectric Patterns? A Brain-Computer Interface with Deep Learning

**J. Laiho** — M.Sc. thesis, Aalto University, August 2020 (with Accenture Liquid Studio, NL)

Part of [Health and biosignals](../areas/health-biosignals.md).

A brain-computer interface reads electrical activity from the scalp and turns it into control
signals, letting someone steer a wheelchair or a cursor by imagining a movement. The obstacle
to deployment is not accuracy in the lab but calibration time. Electroencephalography (EEG)
signals vary so much between people that each new user typically spends weeks training the
system on their own brain. This
thesis asks whether a deep network can cut that cost, and in particular whether it can work on
a person it has never seen.

## The learning problem

**Data points.** One [data point](https://dictionaryofml.org/terms/datapoint.html) is a
single motor-imagery *trial*: a 2-second window, experimentally selected from the 2–4 s
segment of each 8-second trial, recorded across 22 EEG electrodes at 250 Hz. The data is a
public 9-subject benchmark; each subject performed two sessions on different days, six runs
per session, 48 trials per run, with the four classes evenly balanced.

**Features.** Deliberately not hand-engineered. The raw signal is bandpass filtered with a
fifth-order Butterworth filter between 7 and 40 Hz — keeping the alpha and beta bands that
carry motor-imagery information, and cutting the low frequencies where eye movement and
blinks put most of their power. After that the network sees channels × time directly, and
[learns its own features](https://dictionaryofml.org/terms/featlearn.html). Doing feature
extraction and classification in one model is the main design choice of the thesis.

**Labels.** One of four imagined movements: left
hand, right hand, feet or tongue. That makes this a four-way
[classification](https://dictionaryofml.org/terms/classification.html) problem. The model
outputs a probability per class.

**Hypothesis space.** A convolutional neural network, using convolution and pooling to exploit
the temporal and spatial structure across electrodes. A second variant adds
[transfer learning](https://dictionaryofml.org/terms/transferlearning.html): weights from a
[pre-trained](https://dictionaryofml.org/terms/pretraining.html) model are copied in and
frozen, so fewer parameters need fitting and calibration time drops.

**Loss function.** *Not confirmed.* The full text is only available as a scanned PDF without a
text layer, and neither the abstract nor the recorded defence states the training objective,
so it is left unstated here rather than guessed.

## What it found

Within-subject, with the model personalised to one person, mean accuracy was 58% ± 15%, which
pre-training improved by a further 2.4% ± 1.3%. Between subjects, accuracy fell to 35% ± 11%.

Against the 25% expected from guessing on four balanced classes, 35% is better than guessing
but far below usable, and the thesis says so. The gap between 58% and 35% is the main result,
and it illustrates a general problem: the distribution shifts from person to person, so a
model fitted to one brain transfers to the next at 35% accuracy. The weeks of
calibration that motivated the work are reduced by pre-training, not eliminated.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/IouXWsJsm0Q" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

[Read the thesis at Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/46105) (open access)
