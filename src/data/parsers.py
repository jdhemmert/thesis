from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedBiography:
    biography: str
    qa_pairs: list  # list of (question, answer) tuples
    attributes: dict = field(default_factory=dict)


class DataParser(ABC):
    expands_rows: bool = False

    @abstractmethod
    def parse(self, example: dict) -> ParsedBiography:
        ...


class QADataParser(DataParser):
    """Parses datasets with explicit biography/question/answer fields."""

    def parse(self, example: dict) -> ParsedBiography:
        return ParsedBiography(
            biography=example["biography"],
            qa_pairs=[(example["question"], example["answer"])],
        )


class AttributesDataParser(DataParser):
    """Parses structured attribute datasets (PLM or Standard format) into QA pairs."""

    expands_rows: bool = True

    def parse(self, example: dict) -> ParsedBiography:
        is_plm = "first_name" in example and "last_name" in example

        if is_plm:
            name = f"{example['first_name']} {example.get('middle_name', '')} {example['last_name']}".replace("  ", " ").strip()
            bio_text = example.get("bio", "")
            company1name = example.get("company1name", "")
            qa_templates = [
                ("What is the birth city of {name}?", example.get("birthcity", "")),
                ("Which university did {name} attend?", example.get("university", "")),
                ("What is the field of study of {name}?", example.get("field", "")),
                ("What is the job of {name}?", example.get("job", "")),
                ("Which company did {name} work for?", example.get("company1name", "")),
                ("In which city is the company {company1name} located?", example.get("company1city", "")),
                ("In what year was {name} born?", str(example.get("birthyear", ""))),
                ("In what month was {name} born?", example.get("birthmonth", "")),
            ]
        else:
            name = example.get("name", "Unknown")
            bio_text = example.get("biography", "")
            company1name = ""
            qa_templates = [
                ("What is the birth date of {name}?", example.get("birth_date", "")),
                ("What is the birth city of {name}?", example.get("birth_city", "")),
                ("Which university did {name} study?", example.get("college", "")),
                ("What major did {name} study?", example.get("major", "")),
                ("Which company did {name} work for?", example.get("company", "")),
            ]

        qa_pairs = []
        for q_tmpl, a in qa_templates:
            if not a:
                continue
            q = q_tmpl.format(name=name, company1name=company1name)
            qa_pairs.append((q, str(a)))

        return ParsedBiography(
            biography=bio_text,
            qa_pairs=qa_pairs,
            attributes=dict(example),
        )


class WikitextDataParser(DataParser):
    """Parses raw text datasets (e.g. WikiText). No QA pairs available."""

    def parse(self, example: dict) -> ParsedBiography:
        return ParsedBiography(
            biography=example.get("text", ""),
            qa_pairs=[],
        )
