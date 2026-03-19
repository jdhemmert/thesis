from dataclasses import dataclass

@dataclass
class MemoryTaskPreprocessor:
    tokenizer: any
    max_length: int
    prompts: any
    dataset_mode: str

    def __call__(self, examples):
        """
        Builds oracle+memory sequences and computes prompt lengths.
        Supports both QA pairs and raw text (e.g. WikiText).
        """
        max_length = self.max_length

        if self.dataset_mode == "wikitext" or ("text" in examples and "question" not in examples):
            # Raw text mode (e.g. WikiText for Null-Alignment)
            texts = examples["text"]
            oracle_texts = texts
            memory_texts = texts

            oracle_pref_lens = [0] * len(texts)
            memory_pref_lens = [0] * len(texts)
            
            bios, qs, ans = texts, [""] * len(texts), [""] * len(texts)

        elif self.dataset_mode == "attributes":
            # Attributes mode: each row is a biography attribute set
            # Support both plm_bio_attributes and biography_attributes
            
            oracle_texts, memory_texts = [], []
            oracle_prefixes, memory_prefixes = [], []
            bios_out, qs_out, ans_out = [], [], []
            
            batch_size = len(next(iter(examples.values())))
            
            for i in range(batch_size):
                ex = {k: v[i] for k, v in examples.items()}
                
                # Detect format
                is_plm = "first_name" in ex and "last_name" in ex
                
                if is_plm:
                    name = f"{ex['first_name']} {ex['middle_name']} {ex['last_name']}".replace("  ", " ").strip()
                    bio_text = ex.get("bio", "")
                    # Generate Q/A pairs for PLM
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
                    # Generate Q/A pairs for Standard
                    qa_pairs = [
                        ("What is the birth date of {name}?", ex.get("birth_date", "")),
                        ("What is the birth city of {name}?", ex.get("birth_city", "")),
                        ("Which university did {name} study?", ex.get("college", "")),
                        ("What major did {name} study?", ex.get("major", "")),
                        ("Which company did {name} work for?", ex.get("company", "")),
                    ]
                
                for q_tmpl, a in qa_pairs:
                    if not a: continue
                    
                    q = q_tmpl.format(name=name, company1name=ex.get("company1name", ""))
                    
                    oracle_texts.append(self.prompts.contextual_qa_training.format(biography=bio_text, question=q, answer=a))
                    memory_texts.append(self.prompts.direct_qa_training.format(question=q, answer=a))
                    oracle_prefixes.append(self.prompts.contextual_qa_generation.format(biography=bio_text, question=q))
                    memory_prefixes.append(self.prompts.direct_qa_generation.format(question=q))
                    
                    bios_out.append(bio_text)
                    qs_out.append(q)
                    ans_out.append(a)
            
            oracle_pref = self.tokenizer(oracle_prefixes, truncation=True, max_length=max_length, add_special_tokens=True)
            memory_pref = self.tokenizer(memory_prefixes, truncation=True, max_length=max_length, add_special_tokens=True)
            oracle_pref_lens = [len(x) for x in oracle_pref.input_ids]
            memory_pref_lens = [len(x) for x in memory_pref.input_ids]
            
            # Reassign for final tokenization
            bios, qs, ans = bios_out, qs_out, ans_out
        elif self.dataset_mode == "qa":
            # QA mode (standard memory training)
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
            
            oracle_pref = self.tokenizer(oracle_prefixes, truncation=True, max_length=max_length, add_special_tokens=True)
            memory_pref = self.tokenizer(memory_prefixes, truncation=True, max_length=max_length, add_special_tokens=True)
            
            oracle_pref_lens = [len(x) for x in oracle_pref.input_ids]
            memory_pref_lens = [len(x) for x in memory_pref.input_ids]
        else:
            raise RuntimeError(f"Unrecognized dataset_mode: {self.dataset_mode}")
    
        oracle = self.tokenizer(oracle_texts, truncation=True, max_length=max_length, add_special_tokens=True)
        memory = self.tokenizer(memory_texts, truncation=True, max_length=max_length, add_special_tokens=True)
    
        return {
            "oracle_input_ids": oracle.input_ids,
            "oracle_attention_mask": oracle.attention_mask,
            "memory_input_ids": memory.input_ids,
            "memory_attention_mask": memory.attention_mask,
            "oracle_prompt_len": oracle_pref_lens,
            "memory_prompt_len": memory_pref_lens,
            "biography": bios,
            "question": qs,
            "answer": ans,
        }
