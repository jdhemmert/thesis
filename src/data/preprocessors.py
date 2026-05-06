from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class MemoryTaskPreprocessor:
    tokenizer: Any
    max_length: int
    prompts: Any
    dataset_mode: str = "qa"
    padding: Optional[str] = "do_not_pad"
    parser: Optional[Any] = None       # DataParser instance (new path)
    biography_task: Optional[Any] = None  # BiographyTask instance (new path)

    def __call__(self, examples):
        if self.parser is not None and self.biography_task is not None:
            return self._call_new_path(examples)
        return self._call_legacy_path(examples)

    def _call_new_path(self, examples):
        batch_size = len(next(iter(examples.values())))

        oracle_texts, memory_texts = [], []
        oracle_prefixes, memory_prefixes = [], []
        eval_question_texts = []
        bios, qs, ans = [], [], []

        for i in range(batch_size):
            ex = {k: v[i] for k, v in examples.items()}
            parsed = self.parser.parse(ex)
            pairs = self.biography_task.build_pairs(parsed)
            for pair in pairs:
                oracle_texts.append(pair.oracle_text)
                memory_texts.append(pair.memory_text)
                oracle_prefixes.append(pair.oracle_prefix)
                memory_prefixes.append(pair.memory_prefix)
                eval_question_texts.append(pair.eval_question_text)
                bios.append(pair.biography)
                qs.append(pair.question)
                ans.append(pair.answer)

        max_length = self.max_length

        oracle = self.tokenizer(oracle_texts, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
        memory = self.tokenizer(memory_texts, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
        oracle_pref = self.tokenizer(oracle_prefixes, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
        memory_pref = self.tokenizer(memory_prefixes, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
        eval_pref = self.tokenizer(eval_question_texts, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)

        result = {
            "oracle_input_ids": oracle.input_ids,
            "oracle_attention_mask": oracle.attention_mask,
            "memory_input_ids": memory.input_ids,
            "memory_attention_mask": memory.attention_mask,
            "oracle_prompt_len": [sum(m) for m in oracle_pref.attention_mask],
            "memory_prompt_len": [sum(m) for m in memory_pref.attention_mask],
            "eval_oracle_input_ids": oracle_pref.input_ids,
            "eval_oracle_attention_mask": oracle_pref.attention_mask,
            "eval_memory_input_ids": eval_pref.input_ids,
            "eval_memory_attention_mask": eval_pref.attention_mask,
            "biography": bios,
            "question": qs,
            "answer": ans,
        }
        return result

    def _call_legacy_path(self, examples):
        max_length = self.max_length
        oracle_pref = None
        memory_pref = None

        if self.dataset_mode != "chunked_lm" and (self.dataset_mode == "wikitext" or ("text" in examples and "question" not in examples)):
            texts = examples["text"]
            oracle_texts = texts
            memory_texts = texts

            oracle_pref_lens = [0] * len(texts)
            memory_pref_lens = [0] * len(texts)

            bios, qs, ans = texts, [""] * len(texts), [""] * len(texts)

        elif self.dataset_mode == "chunked_lm":
            oracle_texts, memory_texts = [], []
            oracle_pref_lens, memory_pref_lens = [], []

            for text in examples["text"]:
                if len(text) < 100:
                    continue
                mid = len(text) // 2
                first_half = text[:mid]
                second_half = text[mid:]
                oracle_texts.append(text)
                memory_texts.append(second_half)
                first_tok = self.tokenizer(
                    first_half, add_special_tokens=True, truncation=True, max_length=max_length
                ).input_ids
                oracle_pref_lens.append(len(first_tok))
                memory_pref_lens.append(0)

            bios = oracle_texts
            qs   = [""] * len(oracle_texts)
            ans  = memory_texts

        elif self.dataset_mode == "attributes":
            oracle_texts, memory_texts = [], []
            oracle_prefixes, memory_prefixes = [], []
            bios_out, qs_out, ans_out = [], [], []

            batch_size = len(next(iter(examples.values())))

            for i in range(batch_size):
                ex = {k: v[i] for k, v in examples.items()}

                is_plm = "first_name" in ex and "last_name" in ex

                if is_plm:
                    name = f"{ex['first_name']} {ex['middle_name']} {ex['last_name']}".replace("  ", " ").strip()
                    bio_text = ex.get("bio", "")
                    qa_pairs = [
                        ("What is the birth city of {name}?", ex.get("birthcity", "")),
                        ("Which university did {name} attend?", ex.get("university", "")),
                        ("What is the field of study of {name}?", ex.get("field", "")),
                        ("What is the job of {name}?", ex.get("job", "")),
                        ("Which company did {name} work for?", ex.get("company1name", "")),
                        ("In which city is the company {company1name} located?", ex.get("company1city", "")),
                        ("In what year was {name} born?", str(ex.get("birthyear", ""))),
                        ("In what month was {name} born?", ex.get("birthmonth", "")),
                    ]
                else:
                    name = ex.get("name", "Unknown")
                    bio_text = ex.get("biography", "")
                    qa_pairs = [
                        ("What is the birth date of {name}?", ex.get("birth_date", "")),
                        ("What is the birth city of {name}?", ex.get("birth_city", "")),
                        ("Which university did {name} study?", ex.get("college", "")),
                        ("What major did {name} study?", ex.get("major", "")),
                        ("Which company did {name} work for?", ex.get("company", "")),
                    ]

                for q_tmpl, a in qa_pairs:
                    if not a:
                        continue
                    q = q_tmpl.format(name=name, company1name=ex.get("company1name", ""))
                    oracle_texts.append(self.prompts.contextual_qa_training.format(biography=bio_text, question=q, answer=a))
                    memory_texts.append(self.prompts.direct_qa_training.format(question=q, answer=a))
                    oracle_prefixes.append(self.prompts.contextual_qa_generation.format(biography=bio_text, question=q))
                    memory_prefixes.append(self.prompts.direct_qa_generation.format(question=q))
                    bios_out.append(bio_text)
                    qs_out.append(q)
                    ans_out.append(a)

            oracle_pref = self.tokenizer(oracle_prefixes, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
            memory_pref = self.tokenizer(memory_prefixes, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
            oracle_pref_lens = [sum(m) for m in oracle_pref.attention_mask]
            memory_pref_lens = [sum(m) for m in memory_pref.attention_mask]
            bios, qs, ans = bios_out, qs_out, ans_out

        elif self.dataset_mode == "qa":
            bios = examples["biography"]
            qs   = examples["question"]
            ans  = examples["answer"]

            oracle_texts = [
                self.prompts.contextual_qa_training.format(biography=b, question=q, answer=a)
                for b, q, a in zip(bios, qs, ans)
            ]
            memory_texts = [
                self.prompts.direct_qa_training.format(question=q, answer=a)
                for q, a in zip(qs, ans)
            ]
            oracle_prefixes = [
                self.prompts.contextual_qa_generation.format(biography=b, question=q)
                for b, q in zip(bios, qs)
            ]
            memory_prefixes = [
                self.prompts.direct_qa_generation.format(question=q)
                for q in qs
            ]

            oracle_pref = self.tokenizer(oracle_prefixes, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
            memory_pref = self.tokenizer(memory_prefixes, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
            oracle_pref_lens = [sum(m) for m in oracle_pref.attention_mask]
            memory_pref_lens = [sum(m) for m in memory_pref.attention_mask]

        else:
            raise RuntimeError(f"Unrecognized dataset_mode: {self.dataset_mode}")

        oracle = self.tokenizer(oracle_texts, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)
        memory = self.tokenizer(memory_texts, truncation=True, max_length=max_length, add_special_tokens=True, padding=self.padding)

        eval_memory = memory_pref if memory_pref is not None else memory
        eval_oracle = oracle_pref if oracle_pref is not None else oracle

        return {
            "oracle_input_ids": oracle.input_ids,
            "oracle_attention_mask": oracle.attention_mask,
            "memory_input_ids": memory.input_ids,
            "memory_attention_mask": memory.attention_mask,
            "oracle_prompt_len": oracle_pref_lens,
            "memory_prompt_len": memory_pref_lens,
            "eval_oracle_input_ids": eval_oracle.input_ids,
            "eval_oracle_attention_mask": eval_oracle.attention_mask,
            "eval_memory_input_ids": eval_memory.input_ids,
            "eval_memory_attention_mask": eval_memory.attention_mask,
            "biography": bios,
            "question": qs,
            "answer": ans,
        }
