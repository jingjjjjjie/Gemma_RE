import json
import os

PROJECT_ROOT = '/root/autodl-tmp/Projects/Gemma_RE/'
input_filename = 'semeval/semeval_val.txt'
output_filename = '/root/autodl-tmp/Projects/Gemma_RE/gemma_re_y/semeval/semeval_val_processed.jsonl'
relation_dict = 'semeval/semeval_rel2id.json'

input_file_path = os.path.join(PROJECT_ROOT, input_filename)
relation_dict_path = os.path.join(PROJECT_ROOT, relation_dict)

def process_training_data_and_save(txt_filepath, dic, output_filepath): 
    relation_dict = {}
    with open(dic, 'r') as jsonl_file:
        for line in jsonl_file:
            relation_data = json.loads(line)
            relation_dict.update(relation_data)

    with open(txt_filepath, 'r') as txt_file, open(output_filepath, 'w') as output_file:
        for line in txt_file:
            data = json.loads(line)
            processed_example = {
                "prompt": " ".join(data["token"]),
                "completion": 
                {
                    "head": data['h']['name'],
                    "tail": data['t']['name'],
                    "relation": data['relation']
                }
            }
            
            output_file.write(json.dumps(processed_example) + '\n')


process_training_data_and_save(input_file_path, relation_dict_path, output_filename)