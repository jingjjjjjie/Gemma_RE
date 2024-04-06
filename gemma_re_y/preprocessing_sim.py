import json
import os

PROJECT_ROOT = '/root/autodl-tmp/Projects/Gemma_RE/'
input_filename = 'semeval/semeval_test.txt'
output_filename = 'semeval/semeval_test_processed_sim.jsonl'
relation_dict = 'semeval/semeval_rel2id.json'

input_file_path = os.path.join(PROJECT_ROOT, input_filename)
output_file_path = os.path.join(PROJECT_ROOT, output_filename)
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
                "completion": {
                    "h": data['h']['name'],
                    "hpos": f"({data['h']['pos'][0]}, {data['h']['pos'][1]})",
                    "t": data['t']['name'],
                    "tpos": f"({data['t']['pos'][0]}, {data['t']['pos'][1]})",
                    "relation": data['relation']
                }
            }
            
            output_file.write(json.dumps(processed_example) + '\n')


process_training_data_and_save(input_file_path, relation_dict_path, output_file_path)