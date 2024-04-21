# Gemma7b LLM Finetuned for Entity Extraction and Classification Task
知识图谱与信息表示'24 HW1

## Description

### Trained Models Overview

We have finetuned Gemma7b with QLoRa on a series of models for entity extraction and classification capabilities:

- **gemma_re_x**: This is the first model we trained. It was developed without dataset packing, which may result in the output of extra irrelevant characters.

- **gemma_re_y**: Our second model was specifically trained with dataset packing. It is optimized to avoid producing irrelevant characters, focusing solely on delivering relevant outputs.

- **gemma_re_z**: This model is dedicated to relation classification tasks and does not perform entity extraction. It operates with entities that are already labeled in the input, focusing strictly on classifying relationships.


## Usage:
The training files and evaluation files are present in each model's folder, under the notebook directory.
