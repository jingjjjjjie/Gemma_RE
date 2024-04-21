# Project Title

Gemma7b LLM Finetuned for entity extraction and classification task 

## Description

Trained Models:
gemma_re_x : first model trained without dataset packing, will output extra irrelevant characters.
gemma_re_y : second model trained with dataset packing, optimized to avoid producing irrelevant characters, focusing solely on relevant outputs.
gemma_re_z : trained only for relation classification without performing entity extraction (entity labeled in the input).
