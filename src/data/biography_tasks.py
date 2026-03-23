from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.data.parsers import ParsedBiography
from src.data.masking import AttributeMasker, Masker


@dataclass
class OracleMemoryPair:
    oracle_text: str        # full sequence including answer tokens
    memory_text: str        # full sequence including answer tokens
    oracle_prefix: str      # everything before the answer — used to compute prompt_len
    memory_prefix: str
    biography: str          # passed through as metadata for extrinsic eval
    question: str = ""
    answer: str = ""
    eval_question_text: str = ""  # QA-format generation prompt for extrinsic eval


@dataclass
class TaskPromptConfig:
    """
    Prompt templates for a single biography task.

    Each template is used in two forms:
      - *_training:   full sequence including the answer (for teacher-forcing)
      - *_generation: prefix only, up to but not including the answer (for prompt_len
                      computation and at inference time)

    The contextual_* variants include the full biography; the direct_* variants do not
    (these are the memory stream inputs).
    """
    contextual_training: str
    contextual_generation: str
    direct_training: str
    direct_generation: str


_QA_PROMPTS = TaskPromptConfig(
    contextual_training="Biography: {biography}\nQuestion: {question}\nAnswer: {answer}",
    contextual_generation="Biography: {biography}\nQuestion: {question}\nAnswer:",
    direct_training="Question: {question}\nAnswer: {answer}",
    direct_generation="Question: {question}\nAnswer:",
)

_CLOZE_PROMPTS = TaskPromptConfig(
    contextual_training="Biography: {biography}\nFill in the blanks: {masked_biography}\nAnswer: {spans}",
    contextual_generation="Biography: {biography}\nFill in the blanks: {masked_biography}\nAnswer:",
    direct_training="Fill in the blanks: {masked_biography}\nAnswer: {spans}",
    direct_generation="Fill in the blanks: {masked_biography}\nAnswer:",
)


def _resolve_prompts(prompts, default: TaskPromptConfig) -> TaskPromptConfig:
    """
    Normalise a prompts argument that may be None, a TaskPromptConfig, or a
    Hydra DictConfig / plain dict (both support attribute access via .field).
    """
    if prompts is None:
        return default
    if isinstance(prompts, TaskPromptConfig):
        return prompts
    # Hydra DictConfig or plain dict
    return TaskPromptConfig(**{k: prompts[k] for k in
                               ("contextual_training", "contextual_generation",
                                "direct_training", "direct_generation")})


class BiographyTask(ABC):
    @abstractmethod
    def build_pairs(self, parsed: ParsedBiography) -> list:
        """
        Build a list of OracleMemoryPair from a parsed biography.
        Returns an empty list if no valid pairs can be constructed.
        """
        ...


class QABiographyTask(BiographyTask):
    """
    Standard QA task: oracle sees biography + question, memory sees question only.
    One pair per QA pair in the parsed biography.
    """

    def __init__(self, prompts=None):
        self.prompts = _resolve_prompts(prompts, _QA_PROMPTS)

    def build_pairs(self, parsed: ParsedBiography) -> list:
        p = self.prompts
        pairs = []
        for q, a in parsed.qa_pairs:
            pairs.append(OracleMemoryPair(
                oracle_text=p.contextual_training.format(
                    biography=parsed.biography, question=q, answer=a),
                memory_text=p.direct_training.format(
                    question=q, answer=a),
                oracle_prefix=p.contextual_generation.format(
                    biography=parsed.biography, question=q),
                memory_prefix=p.direct_generation.format(
                    question=q),
                biography=parsed.biography,
                question=q,
                answer=a,
                eval_question_text=p.direct_generation.format(question=q),
            ))
        return pairs


class ClozeBiographyTask(BiographyTask):
    """
    Biography cloze task: oracle sees full bio + masked bio, memory sees masked bio only.

    For each QA pair in the parsed biography, attempts to mask the answer value in
    the biography text. Pairs where the value cannot be located are skipped.

    Training objective: KL between oracle and memory at the span answer positions.
    Extrinsic evaluation: QA-format generation prompt stored in eval_question_text,
    using the direct_generation template from _QA_PROMPTS so it matches the format
    the base model expects for question answering.
    """

    def __init__(self, masker: Masker = None, prompts=None):
        self.masker = masker if masker is not None else AttributeMasker()
        self.prompts = _resolve_prompts(prompts, _CLOZE_PROMPTS)

    def build_pairs(self, parsed: ParsedBiography) -> list:
        p = self.prompts
        pairs = []
        for q, a in parsed.qa_pairs:
            masked_bio, found = self.masker.mask_one(parsed.biography, a)
            if not found:
                continue

            pairs.append(OracleMemoryPair(
                oracle_text=p.contextual_training.format(
                    biography=parsed.biography,
                    masked_biography=masked_bio,
                    spans=a),
                memory_text=p.direct_training.format(
                    masked_biography=masked_bio,
                    spans=a),
                oracle_prefix=p.contextual_generation.format(
                    biography=parsed.biography,
                    masked_biography=masked_bio),
                memory_prefix=p.direct_generation.format(
                    masked_biography=masked_bio),
                biography=parsed.biography,
                question=q,
                answer=a,
                eval_question_text=_QA_PROMPTS.direct_generation.format(question=q),
            ))
        return pairs
