from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class AugmentedGPT2Config(GPT2Config):
    def __init__(
        self,
        virtual_token_count=20,
        insert_layer=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.virtual_token_count = virtual_token_count
        self.insert_layer = insert_layer


class AugmentedGPT2Model(GPT2Model):

    def __init__(self, config: GPT2Config, insert_layer: int = 0, virtual_token_count: int = 0):
        super().__init__(config)
        self.insert_layer = insert_layer
        self.virtual_token_count = None
        self.virtual_prompt = None
        self.rebuild_virtual_prompt(virtual_token_count)

    def rebuild_virtual_prompt(self, new_virtual_token_count: Optional[int] = None, weights: Optional[torch.Tensor] = None):
        if weights is not None:
            new_virtual_token_count = weights.shape[0]

        if new_virtual_token_count is not None:
            self.virtual_token_count = new_virtual_token_count

        if self.virtual_token_count is None:
            self.virtual_token_count = 0

        if self.virtual_token_count > 0:
            dtype = self.wte.weight.dtype
            device = self.wte.weight.device
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
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        virtual_tokens: Optional[torch.FloatTensor] = None,
        use_virtual_tokens: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions | Tuple:
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if use_virtual_tokens or (virtual_tokens is not None and use_virtual_tokens is not False):
            final_virtual_tokens = None
            if virtual_tokens is not None:
                final_virtual_tokens = virtual_tokens
            elif self.virtual_prompt is not None and self.virtual_token_count > 0:
                batch_size = inputs_embeds.shape[0]
                prompt_indices = torch.arange(self.virtual_token_count, device=inputs_embeds.device)
                final_virtual_tokens = self.virtual_prompt(prompt_indices).unsqueeze(0).expand(batch_size, -1, -1)

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
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class AugmentedGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: AugmentedGPT2Config):
        super().__init__(config)

        self.transformer = AugmentedGPT2Model(
            config,
            insert_layer=config.insert_layer,
            virtual_token_count=config.virtual_token_count
        )

        self.post_init()

    @property
    def model(self):
        """Alias for self.transformer to match LlamaForCausalLM's .model interface."""
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_virtual_tokens: Optional[bool] = None,
        **kwargs,
    ):
        # GPT2LMHeadModel.forward() does not forward **kwargs to self.transformer(), so
        # use_virtual_tokens would be silently dropped. Instead, we do the virtual token
        # embedding concat here before delegating to the parent.
        vp = self.transformer.virtual_prompt
        vtc = self.transformer.virtual_token_count
        if use_virtual_tokens is not False and vp is not None and vtc > 0:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.transformer.wte(input_ids)
                input_ids = None

            if inputs_embeds is not None and past_key_values is None:
                batch_size = inputs_embeds.shape[0]
                prompt_indices = torch.arange(vtc, device=inputs_embeds.device)
                virtual = vp(prompt_indices).unsqueeze(0).expand(batch_size, -1, -1)
                inputs_embeds = torch.cat([virtual, inputs_embeds], dim=1)

                if attention_mask is not None:
                    virtual_mask = torch.ones(
                        (batch_size, vtc),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([virtual_mask, attention_mask], dim=1)

        if labels is not None and vp is not None and use_virtual_tokens is not False and vtc > 0:
            batch_size = labels.shape[0]
            padding = torch.full((batch_size, vtc), -100, dtype=torch.long, device=labels.device)
            labels = torch.cat([padding, labels], dim=1)

        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **kwargs
        )
