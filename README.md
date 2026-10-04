# Doing a Thesis in Applied Machine Learning

This repository contains guidance for students doing a **bachelor's, master's, or doctoral thesis** in applied machine learning supervised by [Alex Jung](https://machinelearningforall.github.io/about/). It applies regardless of where you are enrolled. The guide currently covers students at **Aalto University** (Finland) and **IMC Krems University of Applied Sciences** (Austria); everything that depends on your university or degree level is collected in [Thesis Levels](#thesis-levels-bachelors-masters-doctoral) and [University-Specific Information](#university-specific-information), so the rest of the guide reads the same for everyone.

- Supervised theses, all levels and universities: [theses.md](theses.md)
- Open topics: [Topics.md](Topics.md)
- Weekly drop-in session for all thesis students: [Thesis Garage](Garage.md)
- Your university's official thesis rules (templates, forms, deadlines): see [University-Specific Information](#university-specific-information)

---

## Table of Contents

1. [What to Expect from Your Supervisor](#what-to-expect-from-your-supervisor)
2. [What Is Expected from You](#what-is-expected-from-you)
3. [Thesis Levels: Bachelor's, Master's, Doctoral](#thesis-levels-bachelors-masters-doctoral)
4. [Getting Started](#getting-started)
5. [Typical Timeline](#typical-timeline)
6. [Recommended Development Environment](#recommended-development-environment)
7. [Responsible Use of AI](#responsible-use-of-ai)
8. [Practical Workflow](#practical-workflow)
9. [Thesis Manuscript Preparation](#thesis-manuscript-preparation)
10. [Typesetting Mathematical Texts](#typesetting-mathematical-texts)
11. [Iterative Writing Process](#iterative-writing-process)
12. [Self-Editing Pass (Prose Linter)](#self-editing-pass-prose-linter)
13. [Final Thesis Checklist](#final-thesis-checklist)
14. [Thesis Presentation and Self-Evaluation](#thesis-presentation-and-self-evaluation)
15. [Thesis Evaluation, Decision, and Appeals](#thesis-evaluation-decision-and-appeals)
16. [University-Specific Information](#university-specific-information)
17. [References](#references)
18. [Feedback and Questions](#feedback-and-questions)

---

## What to Expect from Your Supervisor

As your supervisor, you can expect me to:

- Help you clearly define your ML problem.
- Advise on suitable ML methods, tools, and resources.
- **Be available weekly in the [Thesis Garage](Garage.md)**, where you can present your progress and get feedback on work in progress.
- Guide you through thesis writing and evaluation.
- Give feedback on **complete, polished drafts** and on your self-assessment.
- Offer the opportunity to discuss your self-assessment before submission.
- Know, or find out with you, which rules of your university apply to your thesis — but the formal process (forms, deadlines, submission system) is run by your programme, not by me.

> **Please note:** I do not review early-stage or partial drafts — for example, a single rough chapter, an unedited first attempt, or notes that have not yet been through your own self-editing pass. I give detailed feedback only once you have a **full draft that you consider finished** and have run through the [self-editing pass](#self-editing-pass-prose-linter) below. Sending work earlier than that does not speed things up; it means feedback gets spent on issues you could have caught yourself. (For a doctoral thesis, "full draft" applies to each article and to the summary separately; see [Thesis Levels](#thesis-levels-bachelors-masters-doctoral).)

## What Is Expected from You

As a thesis student, you are expected to:

- Take ownership of your research and drive progress independently.
- **Drop by the weekly [Thesis Garage](Garage.md) and present your progress regularly** — a result, a plot, or a problem you are stuck on. This is the main place you get feedback on work in progress.
- Come prepared with concrete questions or results.
- Speak up early — tell me as soon as you are stuck or falling behind schedule.
- Use high-quality scientific references (peer-reviewed journals, reputable conferences, established textbooks).
- Follow the writing and typesetting conventions described in this guide.
- Know your own university's thesis regulations (registration, proposal approval, deadlines, formatting template, submission) and keep track of them yourself; see [University-Specific Information](#university-specific-information).

---

## Thesis Levels: Bachelor's, Master's, Doctoral

The writing and methodology conventions in this guide are the same at every level. What changes is the **scope** of the problem, the **depth** of the analysis, and the amount of **originality** expected. The table gives the typical profile; your programme's regulations define the formal requirements (credits, length, deadlines), and those always take precedence.

| | Bachelor's thesis (BSc) | Master's thesis (MSc) | Doctoral thesis (PhD) |
|---|---|---|---|
| **Goal** | Show that you can apply established ML methods correctly to a well-defined problem and report the results in a scientific format | Show that you can independently formulate an ML problem, choose and justify methods, and evaluate them critically | Make an original, peer-reviewed contribution to the state of the art |
| **Typical problem** | A given dataset and a clearly stated task; compare a few standard models against a sensible baseline | A problem you shape yourself (often with an industry partner); design, implement, and evaluate a method, including diagnosis of failure modes | A research programme spanning several related problems, each addressed in a publication |
| **Literature review** | Short: the handful of papers and the textbook chapters that define your methods | Systematic: position your work relative to the most relevant prior work and identify the gap you address | Comprehensive and ongoing; each paper has its own related-work section, the summary ties them together |
| **Typical duration** | A few months alongside courses | 6–12 months, often full-time | 3–5 years |
| **Where this guide applies** | Everything up to and including the [Final Thesis Checklist](#final-thesis-checklist); the [Self-Editing Pass](#self-editing-pass-prose-linter) matters just as much for a short text | All of it | All of it, applied to each article and to the summary; add the `--profile paper` mode of the [linter suite](assets/linters/README.md) for the articles |

**Bachelor's students:** keep the problem small enough that you can finish the full loop — data, model, evaluation, write-up — with time left for revision. A clean, well-reported comparison of standard methods is a good bachelor's thesis; an ambitious method that is half-evaluated is not.

**Master's students:** the distinguishing feature is critical evaluation. Beyond reporting a test score, diagnose *why* a method works or fails (sensitivity analysis, error analysis, comparison to a simple baseline) and relate the findings back to your research questions.

**Doctoral students:** a doctoral thesis is normally **article-based** (a set of peer-reviewed publications plus a summary) or, less commonly, a **monograph**. Each paper is a self-contained piece of work with its own problem formulation, related work, and evaluation; the manuscript conventions below apply to each of them. The summary (also called the compilation part or "introductory chapter") is where this guide's advice on structure and self-contained section openers matters most, because it has to make the papers read as one argument. Doctoral theses are currently supervised at Aalto University; see [Aalto University](#aalto-university) below for the formal requirements of the Doctoral Programme in Science.

---

## Getting Started

To start your thesis:

1. **Formulate your ML problem** clearly by identifying data points, their features, and labels ([watch this video](https://youtu.be/2q5jpvD-638)).
2. **Formulate your research questions** so that each is clear, focused on a single problem, and specific about the metrics involved ([guidance](https://www.monash.edu/library/help/assignments-research/developing-research-questions)). Prefer "to what extent does X improve Y on metric Z" over yes/no forms like "can X beat Y", and do not bundle two questions into one. A bachelor's thesis typically has one or two research questions, a master's thesis two or three.
3. **Choose suitable ML models** that you are comfortable implementing (e.g., linear regression, random forests, or neural networks).
4. **Identify data sources and evaluation criteria** (e.g., test accuracy, computational efficiency).
5. **Check what your programme requires before you start** — most universities require a registered topic and an approved proposal before the work formally counts. See [University-Specific Information](#university-specific-information).

Detailed guidance is available in [Chapter 2 of the textbook *Machine Learning: The Basics*](https://doi.org/10.1007/978-981-16-8193-6) (open access via most university libraries) and in these [lecture videos](https://youtube.com/playlist?list=PLrbn2dGrLJK9zB7pdEd8QOtmC9-eoqoch).

---

## Typical Timeline

A thesis moves through the same phases at every level; the phases are shorter and overlap more in a bachelor's thesis, and repeat once per article in a doctoral thesis.

| Phase | Activities |
|---|---|
| **Problem Definition** | Identify research question, data sources, and evaluation criteria; get the topic and proposal approved by your programme where required |
| **Literature Review** | Survey related work; identify gaps your thesis addresses |
| **Data Collection & Preprocessing** | Gather, clean, and explore your dataset |
| **Modeling** | Implement and train ML models; run baseline experiments |
| **Evaluation & Diagnosis** | Benchmarks, sensitivity analysis, error analysis |
| **Writing** | Draft chapters iteratively; incorporate feedback |
| **Self-Assessment & Presentation** | Complete any evaluation form your programme requires; prepare and deliver the thesis presentation or defence |

Typical overall durations: a few months for a bachelor's thesis, 6–12 months for a master's thesis, 3–5 years for a doctoral thesis (see [Thesis Levels](#thesis-levels-bachelors-masters-doctoral)). Your programme's deadlines, not these estimates, are binding.

---

## Recommended Development Environment

[Visual Studio Code](https://code.visualstudio.com/) is a good choice for thesis work — it supports LaTeX (via the [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) extension), Python, and Jupyter notebooks in one place.

The [Claude Code extension for VS Code](https://marketplace.visualstudio.com/items?itemName=Anthropic.claude-code) integrates an AI assistant directly into your editor. You can use it to:

- Explain or refactor Python/LaTeX code in context
- Get feedback on a selected paragraph or equation
- Generate boilerplate (e.g., plotting code, pseudocode skeletons)
- Ask questions about your codebase without leaving the editor

> **Note:** You are accountable for everything in your thesis; an LLM cannot be. Verify every AI-generated output before you rely on it, and check that your university permits the use before you start (see [Responsible Use of AI](#responsible-use-of-ai)).

---

## Responsible Use of AI

Follow your university's official policy; the rules differ between universities and are stricter for students than for staff. The links are collected under [University-Specific Information](#university-specific-information). The principles below hold everywhere.

### Accountability

You are accountable for the entire content of your thesis — every claim, result, and conclusion. An LLM cannot be: it cannot be examined, cannot answer for a mistake, and cannot be listed as an author. "The AI produced it" excuses nothing. Whatever tools you use, you must be able to defend every statement in your thesis as if you had written it unaided.

### Disclosure

- Disclose when and how you used AI tools, in a dedicated statement — not in the Methods section, which is reserved for research methods.
- Record the tool, version, and settings; online services change frequently, so exact reproduction is rarely possible.
- Include the statement even if you used no AI tools, and say so plainly — with no statement, a reader cannot tell "no AI used" from "AI used but not disclosed".
- If your university prescribes a specific form or wording for the disclosure, use that; the dedicated statement is the minimum.

### Data Protection and Intellectual Property

- Do not upload personal, confidential, or unpublished material to public AI services — this may breach GDPR. Use local or GDPR-compliant tools for sensitive data.
- AI output may reproduce others' work without attribution. Verify every citation independently.

### Integrity-Preserving Uses

AI tools can support your work when used for:

- Proofreading language and grammar
- Brainstorming research directions or experiment designs
- Explaining concepts or summarising background (always verify)
- Generating boilerplate or plotting templates (always review and test)
- Surfacing counterarguments to test your reasoning

Universities treat undisclosed or improper AI use as a breach of good scientific practice, with the same consequences as plagiarism — up to the thesis being declared invalid. When in doubt, ask before you use a tool, not after.

## Practical Workflow

A thesis in machine learning typically involves:

- **Data Collection and Preprocessing** using, e.g., [pandas](https://pandas.pydata.org/).
- **Model Training and Validation** using, e.g., [scikit-learn](https://scikit-learn.org/).
- **Model Diagnosis** using numerical experiments (benchmarks, sensitivity analysis) and, when appropriate, mathematical analysis (generalisation bounds, error analysis, comparison to Bayes' risk).

For academic sources, use:

- **Your university library's discovery service** (access to paywalled journals and e-books; see [University-Specific Information](#university-specific-information) for the link)
- **IEEE Xplore**: https://ieeexplore.ieee.org
- **ACM Digital Library**: https://dl.acm.org
- **Scopus**: https://www.scopus.com
- **Web of Science**: https://www.webofscience.com
- **dblp** (computer science bibliography, for finding the published version of a paper): https://dblp.org

To assess the quality of journals and conferences, consult a publication ranking such as the Finnish **JUFO** classification (https://jfp.csc.fi/jufoportal) or the **CORE** conference ranking (https://portal.core.edu.au/conf-ranks/).

If you are uncertain about a reference's quality, ask me.

---

## Thesis Manuscript Preparation

When preparing your thesis, ensure:

- **Template**: Use the official thesis template of your university and programme where one exists (see [University-Specific Information](#university-specific-information)); the conventions below work within any template.
- **Terminology**: Use terms defined in the [Aalto Dictionary of ML](https://aaltodictionaryofml.github.io/). You are encouraged to reuse its LaTeX source (e.g., TikZ figures).
- **Problem formulation**: State clearly what the data points are and how their features and labels are defined.
- **Loss functions**: Explicitly state the loss function used for training and, separately, for validation or testing.
- **Numerical results**: Present and discuss results thoroughly to answer your research questions.
- **Baselines**: Use appropriate baselines or benchmarks (e.g., [Kaggle competitions](https://kaggle.com)).
- **Structure**: Begin each chapter and section with an introductory paragraph explaining its content and its connection to the rest of the thesis. Where a chapter or section has subsections, the introduction should tell the reader what each subsection covers and how they fit together — stating a theme without mapping it onto the subsections leaves the reader without a map.
- **Equations**: Reference all numbered equations using `\eqref{}`. Only number equations that are referenced in the text; leave unreferenced equations unnumbered.
- **Algorithms**: Present new methods as pseudocode ([see examples](https://www.overleaf.com/learn/latex/Algorithms)).
- **Figures**: Ensure all figures are clear, labelled, and have informative captions ([caption guidelines: Rule 4 of the PLOS "Ten Simple Rules for Better Figures"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)). Set the figure size to its final print width so that tick, axis, and legend text renders at no less than roughly 80% of the body-text size, and check that all plots remain readable in a grayscale printout.
- **References**: Format according to [IEEE guidelines](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf) unless your programme prescribes another style (IMC Krems programmes often prescribe their own; check your programme's formatting guide). Whatever the style, apply it consistently and cite the published, peer-reviewed version of a paper rather than its arXiv preprint whenever one exists (check [dblp](https://dblp.org) or the publisher's site).

For creating effective figures, see Edward Tufte's [The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi).

---

## Typesetting Mathematical Texts

### Display vs Inline Math

- Use **inline math** (`$...$`) for short expressions within a sentence:
  `The loss is defined as $L(\theta)$.`
- Use **display math** (`\[ ... \]` or the `equation` environment) for standalone equations that are central or referenced.

### Punctuation with Displayed Equations

Punctuate displayed math as part of the surrounding sentence:

```latex
The empirical risk is defined as
\[
L(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f(x_i;\theta), y_i).
\]
```

---

## Iterative Writing Process

- While experimenting, keep a separate working notes file with the methods you try, the hyperparameters used, and the results — without worrying about prose, formatting, or citations. Treat it as a lab notebook.
- Start serious writing with the **literature review** or **methodology** chapters. These settle earliest and rarely need rewriting when your results change.
- Hold off on the **results** and **discussion** chapters until your experiments have settled — these chapters often change a lot, so writing them up too early wastes effort.
- Write the **abstract last**. It is the shortest section but depends on everything else being settled.
- **Get feedback as you go through the weekly [Thesis Garage](Garage.md).** The garage — not written draft review — is where you get feedback on work in progress. Present your progress regularly: a result, a plot, a problem you are stuck on, or a short section you want reactions to. Bring specific questions. This is the fastest way to correct course early, and it is available every week; a finished written draft is not a prerequisite for it.
- Also draw on feedback from peers and, where appropriate, LLM-based tools for quick checks between garage sessions.
- Keep in mind that detailed **written** feedback from me comes only on a **complete, polished draft** you consider finished (see [What to Expect from Your Supervisor](#what-to-expect-from-your-supervisor)). Use the weekly garage sessions for everything before that stage.
- Expect and budget for multiple revision rounds before submission. Some programmes (for example the master's programmes at IMC Krems) formally allow a rejected thesis to be sent back for revision only a limited number of times, so the revision rounds should happen *before* the official submission, not after.

---

## Self-Editing Pass (Prose Linter)

Once a draft is near-final, read it through once looking **only** for the recurring prose defects below. Each is cheap to fix yourself and distracting for a reader — or your supervisor — to flag repeatedly. Fixing them before review means feedback can focus on the substance of your work rather than on mechanics.

- **Excessive forward referencing** — "as we will see in Chapter 5" scattered throughout. Present material where the reader needs it; a few pointers are fine, but a habit of them signals disorganised structure. The [linter suite](assets/linters/README.md) flags forward-looking cue phrases (`prose_lint.py`), floats referenced long before they appear (`crossref_forward_lint.py`), and concepts used before they are defined (`forward_ref_lint.py` / `forward_ref_lint_llm.py`) — run these as a first pass.
- **Undefined or re-defined acronyms** — expand each acronym at first use, then use it consistently; do not re-expand it later.
- **Inconsistent terminology** — pick one term per concept (following the [Aalto Dictionary of ML](https://aaltodictionaryofml.github.io/)) and do not silently switch between synonyms (e.g., "data point" vs "sample" vs "instance").
- **Vague quantifiers** — "significantly", "very", "a lot" without a number. Give the figure or drop the word.
- **Uncited claims** — every empirical or historical assertion that is not your own result needs a citation.
- **Dangling references** — "this shows", "it follows" where "this" or "it" has no clear antecedent.
- **Unmotivated sections** — a chapter or section that opens without a sentence stating what it covers and why it belongs here.
- **Openers that lean on the previous section** — readers (and examiners) enter your thesis at headings, so the first sentence after a heading must stand alone. "A difference of almost an order of magnitude illustrates diverging views" — a difference in *what*? Name the antecedent ("Published estimates of tracking's revenue benefit differ by almost an order of magnitude ...") or the opener is opaque to anyone who did not just read the preceding section. Named references ("the harms reviewed in Section 2.2.4") are fine; bare pronouns and unexplained noun phrases pointing across the heading are not. The same applies within a section: consecutive paragraphs must not jump topic without a bridging sentence.
- **Jargon and undefined evaluative claims** — "smoothest convergence", "performs well", "training is stable": define and measure the property, or drop the claim. A nearby number (a hyperparameter value, a sample size) does not quantify the claim itself. Beware evaluative use of words with a precise technical meaning ("smooth" means differentiable).
- **"Statistically significant"** — use only with an actual statistical test. Without one the phrase claims evidence you do not have; write "reduces sampling variability" or report the spread instead.
- **Type, range, and dimension mismatches in formal claims** — a relation must relate objects of the *same kind*. An active-constraint *set* cannot "approximate" a *strategy* (a bound-assignment vector) unless you first state the map between them; a parameter you defined as a probability cannot be set to 10; a dimensionless speedup is not measured in seconds. State the correspondence, keep every value inside its defined range, and drop stray units. The [`type_consistency_lint_llm.py`](assets/linters/README.md) linter flags these as `TYPE-MISMATCH`, `RANGE`, and `DIMENSION` (and `BRIDGE-LOOSE` when a defined map licenses the shorthand — restate the claim over the mapped object).
- **Tense drift** — keep the tense consistent within a passage when describing methods and results.

Reading aloud, or reading one defect type at a time from start to finish, catches far more than a single general pass.

A linter suite covering every defect on this list — and most of the manuscript checklist above — is available in [`assets/linters/`](assets/linters/README.md): run `python3 run_all_linters.py thesis.pdf` on your compiled PDF (or on your LaTeX sources) before every revision round. The scripts need only Python 3 with `pdftotext` or PyMuPDF; see the [linter README](assets/linters/README.md). The `--llm` checks default to Aalto's hosted AI gateway (Aalto network only), but can also run against a **local on-device model** (e.g. `mlx_lm.server` on Apple Silicon, or Ollama) or any OpenAI-compatible endpoint your own university provides, so your unpublished draft never leaves your machine or your institution — see [Data handling](assets/linters/README.md) in the README. Students outside Aalto should use one of those alternatives.

---

## Final Thesis Checklist

Before submitting, verify each item below. Links point to the relevant section of this guide.

- [x] ML problem is precisely formulated: data points, features, and labels are clearly defined (see [Getting Started](#getting-started))
- [x] Loss functions for training and evaluation are explicitly stated (see [Manuscript Preparation](#thesis-manuscript-preparation))
- [x] Methods are clearly described, including pseudocode for new algorithms
- [x] Baselines or benchmarks are included and discussed
- [x] All figures have labelled axes and informative captions
- [x] All numbered equations, tables, and figures are referenced in the text
- [x] Citations are formatted consistently according to IEEE guidelines or the style your programme prescribes
- [x] Acronyms are expanded at first use and terminology is consistent (see [Self-Editing Pass](#self-editing-pass-prose-linter))
- [x] A prose self-editing pass has been done for forward referencing, vague quantifiers, and dangling references (see [Self-Editing Pass](#self-editing-pass-prose-linter))
- [x] Research questions are focused, specific, and not answerable with a bare yes/no (see [Getting Started](#getting-started))
- [x] Solver/library versions, key tolerance values, and hardware are reported for all experiments
- [x] Figure text is legible at print size and in grayscale
- [x] The AI-use statement is included, naming tools and versions (also if no AI tools were used)
- [x] The thesis uses your university's official template and meets its formal requirements (see [University-Specific Information](#university-specific-information))
- [x] Any self-assessment or evaluation form your programme requires is completed (Aalto master's students: [form here](material/Statement_template_CCIS.docx))

---

## Thesis Presentation and Self-Evaluation

After completing the thesis manuscript:

- Complete the self-assessment your programme requires, with explicit references to sections of your thesis. Aalto master's students use the [evaluation form](material/Statement_template_CCIS.docx); if your programme has no form, write a one-page self-assessment along the same lines — it is the basis for our discussion in any case.
- Read the grading criteria of your programme before submission so you know what a high-quality thesis looks like in your examiners' eyes. For Aalto master's theses this is the [grade characterisation PDF](material/GradeCharact.pdf); for other programmes see [University-Specific Information](#university-specific-information).
- Optionally, request a meeting to discuss your self-assessment before submission.
- Prepare your thesis presentation — either live during a garage session or as a recorded video ([see examples](https://youtube.com/playlist?list=PLrbn2dGrLJK8xt7j0tvaL0uMCdrtQ7JY2)). Where your programme requires a formal presentation or oral defence (for example the *kommissionelle Gesamtprüfung* at IMC Krems or a doctoral defence), the garage is a good place to rehearse it.

---

## Thesis Evaluation, Decision, and Appeals

[YouTube overview](https://youtu.be/HWHWAy9sOFk?si=-6sPplFx0Gvj0SFJ) (recorded for Aalto master's students; the principles are the same elsewhere)

### How the grade is decided

- Your thesis is evaluated against the **official criteria of your programme and university**, not against this guide. This guide tells you how to meet those criteria.
- I prepare a written evaluation and, where the procedure provides for one, a **grade proposal** based on these criteria.
- The **final decision is made by your programme or school** following its formal procedure (a degree committee, a programme director, an examination board).
- Study your programme's grading criteria carefully before submission (links under [University-Specific Information](#university-specific-information)).

### Transparency and feedback

- You have the right to **see the evaluation criteria** applied to your thesis and to receive the written evaluation.
- You may request clarification on how your thesis was assessed and how the grade was formed.
- Your **self-assessment** is an important part of this process.

### Appealing a grading decision

- If you believe an error occurred, you can **request a review** of the decision. The exact instrument differs by country: at Aalto it is a request for rectification of the grade, at IMC Krems a complaint about a defect in the assessment procedure. The deadlines are short — **two weeks** at both universities — so act promptly.
- The formal procedures are linked under [University-Specific Information](#university-specific-information).

**Practical advice:** If unsure whether an appeal is appropriate, discuss the evaluation with your supervisor first. Many issues can be resolved without a formal appeal.

---

## University-Specific Information

Everything in this guide that depends on where you are enrolled is collected here. Your university's regulations always take precedence over this guide. If your university is not listed, send me the link to its thesis regulations and I will add a section.

### Aalto University

Theses are supervised in the Department of Computer Science (School of Science, SCI).

| | Where to look |
|---|---|
| **Bachelor's thesis** | Instructions, template, and seminar schedule on your programme's pages in the Aalto Student Guide ([Aalto Bachelor's Programme in Science and Technology](https://www.aalto.fi/en/programmes/aalto-bachelors-programme-in-science-and-technology)); past theses in [Aaltodoc](https://libguides.aalto.fi/c.php?g=410698&p=2797875) |
| **Master's thesis** | [SCI Master's Thesis Guide on MyCourses](https://mycourses.aalto.fi/course/view.php?id=41665) — templates, forms, evaluation rules; start with the "Get oriented" section |
| **Doctoral thesis** | [Doctoral thesis in the Aalto Doctoral Programme in Science](https://www.aalto.fi/en/programmes/aalto-doctoral-programme-in-science/doctoral-thesis) — article-based vs monograph, pre-examination, defence; general information at [Aalto Doctoral Education](https://www.aalto.fi/en/doctoral-education) |
| **Grading criteria (MSc)** | [Grade characterisation](material/GradeCharact.pdf) and the [self-assessment form](material/Statement_template_CCIS.docx) used by the CCIS master's programme |
| **Responsible AI use** | [Responsible use of AI in the research process](https://www.aalto.fi/en/services/responsible-use-of-artificial-intelligence-in-the-research-process) and [Tips for using AI for students](https://www.aalto.fi/en/services/tips-for-using-artificial-intelligence-for-students) |
| **Library** | [Aalto University Library (Primo)](https://primo.aalto.fi/discovery/search?vid=358AALTO_INST:VU1&lang=en); the Finnish [JUFO](https://jfp.csc.fi/jufoportal) ranking for publication quality |
| **Appeals** | [Academic appeals at Aalto University](https://www.aalto.fi/en/applications-instructions-and-guidelines/academic-appeals): a request for rectification must be submitted within **14 days** of being notified of the grade |
| **LLM access for the linters** | The [linter suite](assets/linters/README.md) defaults to the Aalto AI API / LLM Gateway (Aalto network or VPN) |

### IMC Krems University of Applied Sciences

IMC Krems (IMC Fachhochschule Krems, Austria) offers bachelor's and master's programmes; as a university of applied sciences it does not award doctorates. The binding rules are in the university's **Studien- und Prüfungsordnung** (study and examination regulations); the detailed thesis guides it refers to are internal documents available to enrolled students on the eDesktop intranet.

| | Where to look |
|---|---|
| **Regulations** | [Satzungsteil Studien- und Prüfungsordnung](https://www.imc.ac.at/fileadmin-imckrems/user_upload/Downloads/DE/Rectorate/studien-und-pruefungsordnung.pdf) (German; FHR-5-0035). Bachelor's theses: section 3.8; master's theses: section 3.9 |
| **Thesis guides** | *Leitfaden für Bachelorarbeiten und Bachelorprüfungen* (FHM-5-0008), *Leitfaden für Masterarbeiten und Masterprüfungen* (FHR-5-0009), and the formatting guide *Leitfaden für die formale Gestaltung schriftlicher/wissenschaftlicher Arbeiten* (FHM-5-0003) — all on the eDesktop; ask your programme's study services if you cannot find them |
| **Process** | The topic and the written proposal (*Exposé*) must be approved by the programme director before you start; a master's thesis can be submitted for approval at the earliest three months after the proposal was approved. Surveys or data collection may only begin after proposal approval, and must follow the IMC IT policy and data-protection rules |
| **Assessment** | The supervisors write a moderated assessment report (*Gutachten*), which is handed to you together with the grade. A master's thesis that is not approved can be returned for revision at most twice. The thesis is followed by a presentation and oral examination (*kommissionelle Gesamtprüfung*) |
| **Good scientific practice and AI** | Section 3.7 of the regulations: plagiarism, ghostwriting, data falsification, and improper use of AI invalidate the thesis. Students are bound by the student AI guideline *Richtlinie zur Verwendung von KI durch Studierende* (FHR-5-004, on the eDesktop), which is stricter than the [general staff guideline](https://www.imc.ac.at/fileadmin-imckrems/user_upload/PDF/KI-Richtlinie_FHR-1-0092.pdf) |
| **Library** | The campus library and its licensed databases are reached through the eDesktop; the [University for Continuing Education Krems library](https://www.donau-uni.ac.at/en/university/service/library.html) next door is open to the public |
| **Appeals** | The assessment itself cannot be appealed, but a complaint about a defect in how a negatively assessed examination was conducted can be filed with the programme director within **two weeks**; you may inspect your assessment documents within six months of the grade being announced (regulations, sections 3.1 and 3.2; legal basis: [Fachhochschulgesetz](https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=10009895)) |
| **LLM access for the linters** | Run the `--llm` linters against a local model or an endpoint your university provides; the Aalto gateway is not reachable from outside Aalto (see [Data handling](assets/linters/README.md)) |

### Other universities

Theses supervised earlier at TU Wien are listed in [theses.md](theses.md). If you are enrolled elsewhere and would like me to co-supervise, the general parts of this guide apply unchanged; send me your university's thesis regulations with your first email.

---

## References

### ML Fundamentals

- A. Jung, *Machine Learning: The Basics*. Singapore: Springer, 2022. [[DOI]](https://doi.org/10.1007/978-981-16-8193-6)
- A. Jung et al., "The Aalto Dictionary of Machine Learning," Aalto University. [[GitHub]](https://aaltodictionaryofml.github.io/)
- S. Shalev-Shwartz and S. Ben-David, *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press, 2014.
- [Peer-Review Form — Machine Learning Course Project](material/CS_C3240_PeerReview.pdf)
- [Peer-Review Form — Federated Learning Course Project](material/CS_E4740_PeerReview.pdf)

### Writing and Typesetting

- D. Bertsekas, [Ten Simple Rules for Mathematical Writing](https://www.mit.edu/~dimitrib/Ten_Rules.html)
- D. E. Knuth, T. Larrabee, and P. M. Roberts, [Mathematical Writing](https://mirror.gutenberg-asso.fr/tex.loria.fr/typographie/mathwriting.pdf)
- E. Tufte, *The Visual Display of Quantitative Information*, 2nd ed., 2001.
- [AMS Style Guide for Journals](https://www.ams.org/publications/authors/AMS-StyleGuide-online.pdf)
- [IEEE Editing Mathematics Style Guide](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/Editing-Mathematics.pdf)
- [IEEE Editorial Style Manual for Authors](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Editorial-Style-Manual-for-Authors.pdf), 2024

---

## Feedback and Questions

Reach out via:

- **Email:** alex.jung@aalto.fi (for students at every university)
- **LinkedIn:** [linkedin.com/in/aljung/](https://www.linkedin.com/in/aljung/)
- **YouTube:** [@alexjung111](https://www.youtube.com/@alexjung111)
- **GitHub:** Use issues or pull requests on this repository.
