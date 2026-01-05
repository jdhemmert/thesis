import torch
from transformers import AutoTokenizer
from src.models.augmented_llama import AugmentedLlamaForCausalLM
from src.utils.model import prepare_identical_context_embeddings

def run_poc():
    """
    Runs a proof-of-concept experiment to test the AugmentedLlamaModel,
    exploring the impact of adding Gaussian noise to context-initialized
    virtual token embeddings.
    """
    MODEL = "meta-llama/Llama-3.2-3B"
    CONTEXT_TEXT = "Elizabeth Francesca Bradley's birthday is on January 11, 2037. She grew up in Ogdensburg, NY. She is a graduate of Andrews University. She majored in General Business. She works at Transamerica Corporation."
    PROMPT_TEXT = "What is Elizabeth Francesca Bradley's major?"
    NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.5]

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    augmented_model = AugmentedLlamaForCausalLM.from_pretrained(
        MODEL,
        virtual_token_count=1, # Dummy value, will be rebuilt correctly later
        torch_dtype=torch.bfloat16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM
    standard_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Models loaded.")

    print(f"\nContext: '{CONTEXT_TEXT}'")
    print(f"Prompt: '{PROMPT_TEXT}'")

    # Prepare the initial prompt embeddings from the context text
    initial_virtual_prompt_embedding, prompt_embeds, prompt_attention_mask, full_prompt_ids, virtual_token_count = \
        prepare_identical_context_embeddings(CONTEXT_TEXT, PROMPT_TEXT, tokenizer, augmented_model)
    
    # Rebuild the model's internal prompt to the correct size now that we know it
    augmented_model.model.rebuild_virtual_prompt(virtual_token_count)
    print(f"Augmented model rebuilt with correct virtual_token_count: {virtual_token_count}")

    # Get the standard model's output once to use as a baseline for comparison
    print("\nRunning standard forward pass for comparison (once)...")
    with torch.no_grad():
        standard_outputs = standard_model(input_ids=full_prompt_ids)
    standard_next_token_logits = standard_outputs.logits[:, -1, :]
    standard_probs = torch.nn.functional.softmax(standard_next_token_logits, dim=-1)

    print("\nTop 5 next token predictions (standard):")
    standard_top_k_probs, standard_top_k_ids = torch.topk(standard_probs, 5)
    for i in range(5):
        token = tokenizer.decode(standard_top_k_ids[0, i])
        prob = standard_top_k_probs[0, i].item()
        print(f"  - '{token}' (Probability: {prob:.4f})")
    
    # Run the experiment for each configured noise level
    for noise_level in NOISE_LEVELS:
        print(f"\n--- Running experiment with noise_level={noise_level} ---")
        
        # Set the model's virtual prompt weights for this iteration
        if noise_level > 0:
            noise = torch.randn_like(initial_virtual_prompt_embedding.weight.data) * noise_level
            noisy_weights = initial_virtual_prompt_embedding.weight.data + noise
            augmented_model.model.set_virtual_prompt_weights(noisy_weights)
            print(f"Applied Gaussian noise with std_dev={noise_level} to virtual tokens.")
        else:
            # For noise_level 0, use the clean initial embeddings
            augmented_model.model.set_virtual_prompt_weights(initial_virtual_prompt_embedding.weight.data)
        
        # Run the augmented model forward pass
        with torch.no_grad():
            outputs = augmented_model(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_attention_mask,
                # virtual_tokens argument is omitted, so the model uses its internal prompt
            )

        # Process and print the augmented model's output
        next_token_logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_ids = torch.topk(probs, 5)

        print("\nTop 5 next token predictions with context (augmented):")
        for i in range(5):
            token = tokenizer.decode(top_k_ids[0, i])
            prob = top_k_probs[0, i].item()
            print(f"  - '{token}' (Probability: {prob:.4f})")

        # Compare the augmented output to the standard baseline
        epsilon = 1e-12
        kl_divergence = torch.nn.functional.kl_div((standard_probs + epsilon).log(), probs.to(standard_probs.device), reduction='sum')

        print(f"\nKullback-Leibler (KL) Divergence (augmented || standard): {kl_divergence.item():.4f}")
        print(" (Measures how the augmented distribution diverges from the standard reference. Lower is better.)")


if __name__ == "__main__":
    run_poc()