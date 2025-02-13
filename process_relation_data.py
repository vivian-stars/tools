import json
import re
from string import punctuation
 
def is_punctuation(char):
    return (char in punctuation and char not in ['.','-','/','^']) or (char in '，。！？、；：“”‘’￥…（）——【】、{}')
 
 
def is_chinese(char):
    return True if re.match(r'[\u4e00-\u9fff]|[\u3400-\u4dbf]', char) else False
 
def read_jsonl(path):
    data = []  # 存储解析后的JSON对象的列表
 
    with open(path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line.rstrip('\n'))
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line: {line}. Reason: {e}")
    return data



def add_space(text):
    text_list = []
    length = len(text)
    for i,t in enumerate(text):
        if is_chinese(t) or is_punctuation(t) or (t.isdigit() and (i + 1) < length and is_chinese(text[i+1])):
            text_list.append(t + ' ')
        else:
            text_list.append(t)
    return ''.join(text_list)

def process_relid(relations_list):
    id2rel = {}
    rel2id = {}
    for i,rel in enumerate(relations_list):
        id2rel[str(i)] = add_space(rel)
    for key,val in id2rel.items():
        rel2id[val] = int(key)
    return [id2rel,rel2id]

def process_triples(data):
    triples = []
    relations_set = set()
    for item in data:
        text = item['text']
        entities = item['entities']
        relations = item['relations']
        triple = {}
        
        triple['text'] = add_space(text)
        triple_list = []
        for relation in relations:
            from_id = relation['from_id']
            to_id = relation['to_id']
            relation = relation['type']
            relations_set.add(relation)
            for entity in entities:
                if entity['id'] == from_id:
                    subject = entity['label']
                    start_offset = entity['start_offset']
                    end_offset = entity['end_offset']
                    subject_val = text[start_offset:end_offset]
                elif entity['id'] == to_id:
                    object = entity['label']
                    start_offset = entity['start_offset']
                    end_offset = entity['end_offset']
                    object_val = text[start_offset:end_offset]
            triple_list.append([add_space(subject_val),add_space(relation),add_space(object_val)])
        triple['triple_list'] = triple_list
        triples.append(triple)
    return triples,list(relations_set)

def write_to_json(data,path):
    # 写入到文件
    json_str = json.dumps(data, ensure_ascii=False, indent=4)
    with open(path, "w") as f:
        f.write(json_str)
    print('wrtie success')

if __name__=="__main__":
    path1 = 'data/medical/raw_medical/关系_校对.jsonl'
    path2 = 'data/medical/raw_medical/关系_校对后.jsonl'
    data = read_jsonl(path1)
    data2 = read_jsonl(path2)
    data.extend(data2)
    # print(data[0]['text'])
    # print(data[0]['entities'])
    # print(data[0]['relations'])
    # data = preprocess_entities(data)
    triples,relations_list = process_triples(data)
    length = len(triples)
    train_length = int(length * 0.8)
    test_length = int(length * 0.1)
    train = triples[:train_length]
    test = triples[train_length:train_length+test_length]
    dev = triples[train_length+test_length:]
    print(len(train))
    print(len(test))
    print(len(dev))
    rel2id = process_relid(relations_list)
    train_path = 'data/medical/raw_medical/train_triples.json'
    test_path = 'data/medical/raw_medical/test_triples.json'
    dev_path = 'data/medical/raw_medical/dev_triples.json'
    rel2id_path = 'data/medical/raw_medical/rel2id.json'
    write_to_json(train,train_path)
    write_to_json(test,test_path)
    write_to_json(dev,dev_path)
    write_to_json(rel2id,rel2id_path)
    # pos_text = {}
    # for i,item in enumerate(data[0]['text']):
    #     pos_text[i] = item
    # print(pos_text)


