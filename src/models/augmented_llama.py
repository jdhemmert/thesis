from typing import Optional, Tuple, List, Any
import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention
from dataclasses import dataclass, field
from omegaconf import DictConfig

from src.models.prompt_initializers import PromptInitializerName, INITIALIZER_MAP


class AugmentedLlamaConfig(LlamaConfig):
    """
    Configuration for the Augmented Llama model.
    """
    def __init__(
        self,
        virtual_token_count=20,
        insert_layer=0,
        initialization_method: Optional[PromptInitializerName] = None,
        initialization_context_text=None,
        initialization_noise_level=0.0,
        initializer_config: Optional[DictConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.virtual_token_count = virtual_token_count
        self.insert_layer = insert_layer
        self.initialization_method = initialization_method
        self.initialization_context_text = initialization_context_text
        self.initialization_noise_level = initialization_noise_level
        self.initializer_config = initializer_config

class AugmentedLlamaModel(LlamaModel):

    def __init__(self, config: LlamaConfig, insert_layer: int = 0, virtual_token_count: int = 0):
        super().__init__(config)
        self.insert_layer = insert_layer
        self.virtual_token_count = None
        self.virtual_prompt = None
        self.rebuild_virtual_prompt(virtual_token_count)

    def rebuild_virtual_prompt(self, new_virtual_token_count: Optional[int] = None, weights: Optional[torch.Tensor] = None):
        """
        Creates or replaces the virtual prompt embedding layer.
        If new_virtual_token_count is not provided, it's inferred from the weights.
        If weights are provided, they are used to initialize the embedding layer.
        """
        if weights is not None:
            new_virtual_token_count = weights.shape[0]

        if new_virtual_token_count is not None:
            self.virtual_token_count = new_virtual_token_count
        
        if self.virtual_token_count is None:
            self.virtual_token_count = 0

        if self.virtual_token_count > 0:
            dtype = self.embed_tokens.weight.dtype
            device = self.embed_tokens.weight.device
            self.virtual_prompt = nn.Embedding(
                self.virtual_token_count, self.config.hidden_size, dtype=dtype).to(device)
            
            if weights is not None:
                with torch.no_grad():
                    self.virtual_prompt.weight.data.copy_(weights)
        else:
            self.virtual_prompt = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        virtual_tokens: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | Tuple:
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prioritize passed virtual_tokens, fall back to stored tokens
        final_virtual_tokens = None
        if virtual_tokens is not None:
            final_virtual_tokens = virtual_tokens
        elif self.virtual_prompt is not None and self.virtual_token_count > 0:
            batch_size = inputs_embeds.shape[0]
            prompt_indices = torch.arange(self.virtual_token_count, device=inputs_embeds.device)
            final_virtual_tokens = self.virtual_prompt(prompt_indices).unsqueeze(0).expand(batch_size, -1, -1)

        # Only append virtual_tokens on the first pass
        if final_virtual_tokens is not None and past_key_values is None:
            inputs_embeds = torch.cat([final_virtual_tokens, inputs_embeds], dim=1)

            if attention_mask is not None:
                virtual_attention_mask = torch.ones(
                    (attention_mask.shape[0], final_virtual_tokens.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([virtual_attention_mask, attention_mask], dim=1)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class AugmentedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: AugmentedLlamaConfig):
        super().__init__(config)

        self.model = AugmentedLlamaModel(
            config,
            insert_layer=config.insert_layer,
            virtual_token_count=config.virtual_token_count
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def initialize_virtual_prompt(self, tokenizer: Any, method: PromptInitializerName, config: Optional[DictConfig] = None, dataset_path: Optional[str] = None):
        """
        Initializes the virtual prompt using a specified method.
        """
        print(f"Performing soft prompt initialization with method: '{method.value}'...")
        
        initializer_class = INITIALIZER_MAP.get(method.value)
        if not initializer_class:
            raise ValueError(f"Unknown initializer: {method.value}")
        
        initializer = initializer_class(config)
        
        init_kwargs = {
            "main_model": self.model,
            "main_tokenizer": tokenizer,
            "virtual_token_count": self.config.virtual_token_count,
            "dataset_path": dataset_path,
        }
        
        initial_weights = initializer.initialize(**init_kwargs)
        self.model.rebuild_virtual_prompt(weights=initial_weights)
        
        print(f"Initialized soft prompt with {self.model.virtual_token_count} tokens.")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if labels is not None and self.model.virtual_prompt is not None:
            virtual_token_count = self.model.virtual_token_count
            if virtual_token_count > 0:
                batch_size = labels.shape[0]
                padding = torch.full((batch_size, virtual_token_count), -100, dtype=torch.long, device=labels.device)
                labels = torch.cat([padding, labels], dim=1)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

