---
number: 47
author: "J. Rantanen"
title: "Large Language Models in Mental Health Support"
level: MSc
university: Aalto
date: "2024-08"
industry: null
aaltodoc: "https://urn.fi/URN:NBN:fi:aalto-202408255774"
video: "https://youtu.be/c5Z6CwWmr6w"
video_views: 403
keywords: ["large language model", "transformer", "mental health", "GPT"]
area: llms-generative-ai
framing: not-applicable
confidence: high
sources: [fulltext, abstract, video]
review_status: draft
review_notes: >-
  This is a LITERATURE REVIEW — it trains no model and has no dataset, hypothesis space or
  loss function, so the model/data/loss template is deliberately not used here. This page
  is the worked example of the fallback structure for non-modelling theses.
  System names and the count of case studies are taken from the thesis full text
  (supervisor's copy); an earlier draft, built from defence captions that garbled them,
  described four unnamed systems.
dictionary: [llm, transformer, attention, finetuning, pretraining]
---

# Large Language Models in Mental Health Support

**J. Rantanen** — M.Sc. thesis, Aalto University, August 2024

Part of [Large language models and generative AI](../areas/llms-generative-ai.md).

> **A note on this page.** Most theses in this directory are described through the three
> components of a machine learning problem: the data, the model and the loss function.
> This one is a literature review: it trains nothing and fits nothing, so that structure
> would be an empty form. It is described here by its question, its sources and its
> findings instead.

Demand for mental health care exceeds the supply of professionals almost everywhere, and
conversational artificial intelligence (AI) is being proposed to close the gap. The proposals have arrived faster than
the evidence. This thesis surveys what has actually been demonstrated about
[large language models](https://dictionaryofml.org/terms/llm.html) in mental health support,
and what remains assertion.

## The question

*How can large language models (LLMs) be utilized in mental health support?* The question is
examined across efficacy, real user experience, and the ethical, privacy and legal
constraints on deployment.

## What it surveys

The review builds on the [transformer](https://dictionaryofml.org/terms/transformer.html)
architecture and its [attention](https://dictionaryofml.org/terms/attention.html) mechanism,
then works through thirteen case studies. Four of them carry most of the argument:

- **Psy-LLM**, combining a pre-trained LLM with professional Q&A
  and curated psychological articles, intended as a front-end tool for healthcare
  professionals and a screening aid for urgent cases. Evaluated by perplexity (a language-model fit score) and human
  assessment; judged coherent and relevant, but short of human-level performance.
- **ChatCounselor** (Liu et al., 2023),
  [fine-tuned](https://dictionaryofml.org/terms/finetuning.html) on real
  client–psychologist dialogue: 260 hour-long interviews distilled into the Psych8k
  query–answer set and used for instruction tuning of a Vicuna base model. It surpassed open-source baselines and
  approached commercial-model performance, which the review reads as evidence that
  domain-specific fine-tuning of a modest base model beats a larger general one.
- **Diagnosis of Thought prompting**, which decomposes clinical reasoning into subjective
  assessment, contrastive reasoning to surface cognitive distortions, and schema analysis.
  It improved cognitive-distortion detection and, notably, produced rationales human
  evaluators found interpretable.
- **MindfulDiary**, a journalling application with an AI-driven interface and a clinician
  dashboard, tested in a four-week field study with 28 patients with major depressive
  disorder and five psychiatrists — one of the few clinical evaluations in the set.

The review also draws on first-hand accounts posted publicly by users, which is unusual for
a thesis of this kind and gives it access to failure modes that published evaluations tend
not to report.

## What it found

LLMs improve *accessibility* and *immediacy*: they are available at 3 a.m. and have no
waiting list. There is some evidence of reduced anxiety and depression symptoms. But
effectiveness varies sharply with design and implementation, which is another way of saying
the category "LLM for mental health" does not predict much on its own.

The constraints are where the thesis is firmest. Data security and informed consent are
unresolved in a setting where the disclosed material is about as sensitive as it gets, and
the regulatory position is unclear. The review calls for long-term studies and explicit
governance frameworks rather than further capability demonstrations, a conclusion that has
aged well.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/c5Z6CwWmr6w" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

The thesis record is at [Aaltodoc](https://urn.fi/URN:NBN:fi:aalto-202408255774). The full
text is not open access.
