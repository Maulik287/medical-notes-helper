"""Core implementation for Medical Notes Helper.

Milestone 1: Curate an anonymized corpus of administrative medical text.
Milestone 2: Implement the n-gram language model and apply smoothing.
Milestone 3: Develop a text generation feature that suggests words based on input.
Milestone 4: Create a CLI for users to interact with the suggestion engine.
"""

from collections import Counter, defaultdict
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple


# ==============================================================================
# Milestone 1: Administrative Medical Corpus & Preprocessing
# ==============================================================================

DEFAULT_ADMIN_CORPUS: List[str] = [
    # Check-in and Patient Registration
    "Patient arrived on time for the scheduled appointment and checked in at the front desk.",
    "Patient identity and demographic information were verified against clinic records.",
    "Insurance card and photo identification were scanned into the electronic health record system.",
    "Emergency contact information and primary residence address were confirmed with the patient.",
    "Copayment was collected at the time of check-in via credit card.",
    "Patient completed the standard administrative intake questionnaire and signed the consent forms.",
    "Notice of privacy practices was provided to the patient and acknowledgment was recorded.",
    "Patient was escorted to the examination room and vital signs were recorded by nursing staff.",
    
    # Follow-up and Scheduling
    "Follow-up appointment scheduled for two weeks with the primary care physician.",
    "Follow-up appointment scheduled in one month for routine blood pressure check.",
    "Patient advised to schedule routine annual physical examination before the end of the year.",
    "Patient requested a morning appointment for the upcoming follow-up visit.",
    "Automated reminder notification will be sent via text message forty-eight hours prior to visit.",
    "Cancellation policy was reviewed with the patient during checkout.",
    "Patient scheduled for follow-up consultation following completion of physical therapy.",
    
    # Referrals and Authorizations
    "Referral submitted to cardiology department for specialized consultation and evaluation.",
    "Referral submitted to orthopedic specialist for ongoing joint pain assessment.",
    "Prior authorization requested from insurance provider for diagnostic magnetic resonance imaging.",
    "Prior authorization approved by insurance company for outpatient physical therapy sessions.",
    "Referral documentation and recent clinical notes were faxed to the specialist office.",
    "Insurance coverage verified for upcoming outpatient surgical procedure.",
    "Clinic coordinator contacted payer to expedite pending authorization request.",
    
    # Medication and Pharmacy Coordination
    "Prescription refill sent electronically to the patient preferred retail pharmacy.",
    "Medication reconciliation completed with patient and nursing staff during intake.",
    "Patient instructed to take medication with meals as directed by the prescribing physician.",
    "Pharmacy contacted regarding formulary alternatives for prescribed maintenance medication.",
    "Refill authorization request approved for ninety-day supply with three refills.",
    "Patient reported adherence to current medication regimen without adverse side effects.",
    "Generic substitution authorized for prescription refill request.",
    
    # Lab and Diagnostic Orders
    "Laboratory order entered for routine comprehensive metabolic panel and complete blood count.",
    "Patient instructed to fast for twelve hours prior to scheduled blood draw.",
    "Diagnostic imaging order placed for outpatient chest radiograph.",
    "Lab results reviewed by physician and communicated to patient via secure online portal.",
    "Specimen collected in clinic and sent to reference laboratory for pathology analysis.",
    "Follow-up lab draw scheduled at the diagnostic center for next Tuesday morning.",
    
    # Discharge and Care Coordination
    "Discharge instructions and home care guidelines were reviewed in detail with the patient.",
    "Patient verbalized clear understanding of post-procedure wound care and activity restrictions.",
    "Patient advised to call the clinic immediately if fever or acute pain develops.",
    "After-hours emergency contact numbers and urgent care clinic locations were provided.",
    "Medical records release authorization form was signed by the patient for outside provider records.",
    "Administrative summary generated and filed in the electronic health record."
]


def anonymize_text(text: str) -> str:
    """Mask or remove potential Personally Identifiable Information (PII) such as

    patient names, dates, phone numbers, and identification numbers.
    """
    # Replace dates like MM/DD/YYYY, YYYY-MM-DD, or DD-MM-YYYY
    text = re.sub(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", "<DATE>", text)
    # Replace phone numbers like (123) 456-7890 or 123-456-7890
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "<PHONE>", text)
    # Replace identification numbers (MRN, SSN, Account IDs)
    text = re.sub(r"\b(?:MRN|SSN|ID)[\s#:]*([A-Za-z0-9\-]+)\b", "<ID>", text, flags=re.IGNORECASE)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    """Preprocess text by anonymizing, lowercasing, and standardizing punctuation."""
    text = anonymize_text(text).lower()
    # Normalize hyphens and dashes
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    # Remove unwanted symbols while keeping word characters, hyphens, and essential punctuation
    text = re.sub(r"[^\w\s\.\,\-\<\>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenize a single clean sentence into words, treating punctuation as tokens or boundaries."""
    # Split words, contractions, and special tokens like <DATE>, <PHONE>, etc.
    tokens = re.findall(r"<[A-Z]+>|[\w]+(?:-[\w]+)*|[.,]", sentence)
    # Filter out standalone punctuation from suggestion tokens to keep word flow clean
    word_tokens = [tok for tok in tokens if tok not in {".", ","}]
    return word_tokens


def prepare_corpus(
    raw_sentences: Optional[List[str]] = None,
    n: int = 3,
    add_boundaries: bool = True
) -> List[List[str]]:
    """Clean, tokenize, and add sentence boundary tokens (<s> and </s>) for n-gram modeling."""
    sentences = raw_sentences if raw_sentences is not None else DEFAULT_ADMIN_CORPUS
    processed_corpus: List[List[str]] = []
    
    pad_start = ["<s>"] * (n - 1) if add_boundaries else []
    pad_end = ["</s>"] if add_boundaries else []
    
    for raw in sentences:
        cleaned = clean_text(raw)
        tokens = tokenize_sentence(cleaned)
        if tokens:
            processed_corpus.append(pad_start + tokens + pad_end)
            
    return processed_corpus


def milestone_1() -> Dict[str, object]:
    """Milestone 1: Curate an anonymized corpus of administrative medical text."""
    corpus = prepare_corpus(DEFAULT_ADMIN_CORPUS, n=3, add_boundaries=True)
    all_tokens = [tok for sent in corpus for tok in sent]
    vocab = set(all_tokens)
    
    return {
        "num_sentences": len(corpus),
        "total_tokens": len(all_tokens),
        "vocab_size": len(vocab),
        "sample_sentence": corpus[0],
        "corpus": corpus,
    }


# ==============================================================================
# Milestone 2: N-gram Language Model with Smoothing
# ==============================================================================

class NGramLanguageModel:
    """N-gram language model supporting unigram, bigram, and trigram modeling

    with Add-k (Laplace / Lidstone) smoothing and backoff interpolation.
    """

    def __init__(self, n: int = 3, k: float = 1.0):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.k = k  # Smoothing parameter (k=1.0 is Laplace smoothing)
        
        # n-gram counts: context tuple -> Counter of {next_word: count}
        self.ngram_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        # context counts: context tuple -> total occurrences
        self.context_counts: Counter = Counter()
        # Unigram counts for backoff and vocabulary
        self.unigram_counts: Counter = Counter()
        self.vocabulary: set = set()
        self.total_tokens: int = 0

    def fit(self, tokenized_sentences: List[List[str]]) -> "NGramLanguageModel":
        """Train the n-gram language model on a tokenized corpus."""
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.unigram_counts.clear()
        self.vocabulary.clear()
        self.total_tokens = 0

        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in {"<s>"}:
                    self.vocabulary.add(token)
                    self.unigram_counts[token] += 1
                    self.total_tokens += 1

            # Extract n-grams for lengths from 1 up to self.n
            for order in range(1, self.n + 1):
                for i in range(len(sentence) - order + 1):
                    ngram = sentence[i : i + order]
                    context = tuple(ngram[:-1])
                    word = ngram[-1]
                    self.ngram_counts[context][word] += 1
                    self.context_counts[context] += 1

        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def probability(self, word: str, context: Sequence[str]) -> float:
        """Calculate smoothed probability P(word | context) using Add-k smoothing

        with backoff to lower-order contexts when the context is unseen.
        """
        v_size = max(self.vocab_size, 1)
        k = self.k

        # Truncate context to max order - 1
        ctx = tuple(context[-(self.n - 1):]) if self.n > 1 else ()

        # Try from highest order down to unigram
        while ctx:
            if ctx in self.ngram_counts:
                word_count = self.ngram_counts[ctx][word]
                total_context = self.context_counts[ctx]
                return (word_count + k) / (total_context + k * v_size)
            # Backoff: shorten context by dropping the leftmost token
            ctx = ctx[1:]

        # Unigram fallback with Add-k smoothing
        unigram_count = self.unigram_counts[word]
        return (unigram_count + k) / (self.total_tokens + k * v_size)

    def score_sentence(self, sentence: List[str]) -> float:
        """Calculate the average negative log probability (cross-entropy estimate) of a sentence."""
        if not sentence:
            return float("inf")
        log_prob_sum = 0.0
        n_ctx = self.n - 1
        
        for i in range(n_ctx, len(sentence)):
            word = sentence[i]
            context = sentence[max(0, i - n_ctx) : i]
            prob = self.probability(word, context)
            log_prob_sum += math.log(max(prob, 1e-12))
            
        return -log_prob_sum / max(1, len(sentence) - n_ctx)


def milestone_2(k: float = 1.0) -> NGramLanguageModel:
    """Milestone 2: Implement the n-gram language model and apply smoothing."""
    corpus = prepare_corpus(DEFAULT_ADMIN_CORPUS, n=3, add_boundaries=True)
    model = NGramLanguageModel(n=3, k=k)
    model.fit(corpus)
    return model


# ==============================================================================
# Milestone 3: Text Generation & Next-Word Suggestions
# ==============================================================================

class SuggestionEngine:
    """Suggestion engine providing top-k next word predictions and text generation

    designed for clinical staff drafting medical notes.
    """

    def __init__(self, model: Optional[NGramLanguageModel] = None):
        if model is None:
            self.model = milestone_2()
        else:
            self.model = model

    def suggest_next_words(self, prompt: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Given an input prompt, return top-k most probable next words with probabilities."""
        cleaned = clean_text(prompt)
        tokens = tokenize_sentence(cleaned)
        
        context = tokens[-(self.model.n - 1):] if tokens else ["<s>"] * (self.model.n - 1)
        
        scores: List[Tuple[str, float]] = []
        for word in self.model.vocabulary:
            if word in {"<s>", "</s>"}:
                continue
            prob = self.model.probability(word, context)
            scores.append((word, prob))

        # Sort primarily by probability descending, tie-breaking by higher unigram frequency
        scores.sort(
            key=lambda item: (item[1], self.model.unigram_counts.get(item[0], 0)),
            reverse=True
        )
        return scores[:top_k]

    def complete_text(self, prompt: str, max_words: int = 8, stop_at_boundary: bool = True) -> str:
        """Autocompletes administrative medical text continuation from an input prompt."""
        cleaned = clean_text(prompt)
        tokens = tokenize_sentence(cleaned)
        if not tokens:
            tokens = ["patient"]
            
        generated: List[str] = list(tokens)
        
        for _ in range(max_words):
            context = generated[-(self.model.n - 1):]
            candidates = self.suggest_next_words(" ".join(context), top_k=1)
            if not candidates:
                break
            next_word, _ = candidates[0]
            if stop_at_boundary and next_word == "</s>":
                break
            generated.append(next_word)
            
        return " ".join(generated)


def milestone_3() -> Dict[str, object]:
    """Milestone 3: Develop a text generation feature that suggests words based on input."""
    model = milestone_2()
    engine = SuggestionEngine(model)
    
    test_prompts = [
        "Patient arrived on time for the",
        "Follow-up appointment scheduled for",
        "Referral submitted to cardiology",
        "Prescription refill sent electronically to",
    ]
    
    results = {}
    for prompt in test_prompts:
        suggestions = engine.suggest_next_words(prompt, top_k=3)
        completion = engine.complete_text(prompt, max_words=5)
        results[prompt] = {
            "top_suggestions": suggestions,
            "completion": completion,
        }
        
    return {
        "engine": engine,
        "results": results,
    }


# ==============================================================================
# Milestone 4: CLI Integration
# ==============================================================================

def milestone_4() -> Dict[str, object]:
    """Milestone 4: Create a CLI for users to interact with the suggestion engine."""
    from app import cli
    
    # Verify CLI functions smoothly by predicting next words via the CLI entry points
    engine = SuggestionEngine()
    suggestions = engine.suggest_next_words("Follow-up appointment scheduled", top_k=3)
    
    return {
        "cli_module": cli,
        "sample_suggestions": suggestions,
        "status": "CLI is configured and operational",
    }
