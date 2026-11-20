"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest
from app.core import (
    DEFAULT_ADMIN_CORPUS,
    NGramLanguageModel,
    SuggestionEngine,
    anonymize_text,
    clean_text,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    prepare_corpus,
    tokenize_sentence,
)
from app.cli import main


# Milestone 1: Curate an anonymized corpus of administrative medical text.
def test_milestone_1():
    # 1. Test anonymization of PII
    raw_sample = "Patient visited on 12/04/2026, call 555-234-5678, MRN: A-98432 for appointment."
    anonymized = anonymize_text(raw_sample)
    assert "<DATE>" in anonymized
    assert "<PHONE>" in anonymized
    assert "<ID>" in anonymized
    assert "12/04/2026" not in anonymized
    assert "555-234-5678" not in anonymized
    assert "A-98432" not in anonymized

    # 2. Test tokenization & cleaning
    cleaned = clean_text("Patient arrived for follow-up.")
    tokens = tokenize_sentence(cleaned)
    assert tokens == ["patient", "arrived", "for", "follow-up"]

    # 3. Test milestone_1 curation execution
    result = milestone_1()
    assert result["num_sentences"] >= 30
    assert result["total_tokens"] > 200
    assert result["vocab_size"] > 100
    assert "<s>" in result["sample_sentence"]
    assert "</s>" in result["sample_sentence"]


# Milestone 2: Implement the n-gram language model and apply smoothing.
def test_milestone_2():
    # 1. Train model via milestone_2
    model = milestone_2(k=1.0)
    assert isinstance(model, NGramLanguageModel)
    assert model.vocab_size > 100
    assert model.total_tokens > 200

    # 2. Check counts for known transitions
    # In corpus: "Patient arrived on time for the scheduled appointment..."
    context = ("for", "the")
    prob_known = model.probability("scheduled", context)
    assert prob_known > 0.0

    # 3. Check smoothing for unseen transitions / unknown words
    # An unseen transition must have non-zero probability due to Add-k (Laplace) smoothing
    prob_unseen = model.probability("unseenwordxyz", context)
    assert prob_unseen > 0.0

    # 4. Check that probabilities over the vocabulary sum to ~1.0
    total_prob = sum(model.probability(w, context) for w in model.vocabulary)
    assert 0.8 < total_prob < 1.2

    # 5. Check scoring of sentences
    score_valid = model.score_sentence(["<s>", "<s>", "patient", "arrived", "on", "time", "</s>"])
    assert score_valid < float("inf")


# Milestone 3: Develop a text generation feature that suggests words based on input.
def test_milestone_3():
    # 1. Test next-word suggestion
    engine = SuggestionEngine()
    suggestions = engine.suggest_next_words("Patient arrived on time for the", top_k=3)
    assert len(suggestions) == 3
    suggested_words = [w for w, _ in suggestions]
    assert "scheduled" in suggested_words
    top_word, top_prob = suggestions[0]
    assert top_word == "scheduled"
    assert top_prob > 0.0

    # 2. Test text completion
    completion = engine.complete_text("Follow-up appointment scheduled for", max_words=4)
    assert completion.startswith("follow-up appointment scheduled for")
    assert len(completion.split()) > len("Follow-up appointment scheduled for".split())

    # 3. Test milestone_3 output structure
    m3_result = milestone_3()
    assert "results" in m3_result
    assert "Patient arrived on time for the" in m3_result["results"]


# Milestone 4: Create a CLI for users to interact with the suggestion engine.
def test_milestone_4(capsys):
    # 1. Test milestone_4 function
    m4_result = milestone_4()
    assert "status" in m4_result

    # 2. Test CLI 'suggest' subcommand
    main(["suggest", "Patient arrived on time for the", "--top-k", "2"])
    out = capsys.readouterr().out
    assert "Suggestions for 'Patient arrived on time for the':" in out
    assert "scheduled" in out

    # 3. Test CLI 'complete' subcommand
    main(["complete", "Follow-up appointment scheduled for", "--max-words", "3"])
    out = capsys.readouterr().out
    assert "Completed note:" in out

    # 4. Test CLI 'curate' subcommand
    main(["curate", "--show-samples"])
    out = capsys.readouterr().out
    assert "Administrative Medical Corpus:" in out
    assert "Sentences:" in out
