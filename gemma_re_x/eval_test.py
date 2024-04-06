import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
login('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')


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the checkpoint directory
checkpoint_path = "/root/autodl-tmp/Projects/JJ_ckpt/modelx/checkpoint-2000"  # Replace with your checkpoint path

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path,
                                             quantization_config=bnb_config,
                                             device_map="cuda:0"
                                             )


question = "he directed his criticism at media coverage of the catholic church ."
formatted_prompt = f"### Question: {question}\n ### Answer:"
input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt').to(device)

outputs = model.generate(input_ids, max_length=60, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
