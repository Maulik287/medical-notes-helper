# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2dkzef9fnc/badge.svg)](https://ask.thili.ai/b/2dkzef9fnc)

The Medical Notes Helper is a CLI tool designed for clinic staff in the Healthcare domain. It assists in drafting administrative medical notes by offering next-word suggestions, leveraging an n-gram language model trained on anonymized medical text. This helps ensure consistency and reduces typing effort for users.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps clinic staff draft administrative sections of medical notes by generating next-word suggestions. This tool will provide context-aware suggestions, ensuring consistency and reducing manual typing.

**Done when.** You will have curated an anonymized corpus of administrative medical text, implemented an n-gram language model with smoothing, developed a text generation feature that suggests words based on input, and created a CLI for users to interact with the suggestion engine.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Curate an anonymized corpus of administrative medical text.**  
      Hint: Focus on text preprocessing and tokenization to prepare your corpus.
- [x] **2. Implement the n-gram language model and apply smoothing.**  
      Hint: Build n-gram counts and probabilities, then apply a chosen smoothing technique.
- [x] **3. Develop a text generation feature that suggests words based on input.**  
      Hint: Use your n-gram model to predict the most likely next word given a sequence of input words.
- [x] **4. Create a CLI for users to interact with the suggestion engine.**  
      Hint: Design a command-line interface to take user input and display the generated word suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the corpus with more diverse administrative medical notes to improve suggestion quality.
- Refine the smoothing technique used in the n-gram model for better prediction accuracy.
- Add a feature to allow users to select from multiple next-word suggestions presented in the CLI.
