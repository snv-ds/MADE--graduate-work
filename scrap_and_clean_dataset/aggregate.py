import sys
import json
from tqdm import tqdm


def aggr(file_name):
    with open(file_name, 'r') as f:
        res = f.readlines()
    for row in res:
        yield row.strip()


def get_contents(json_file_name):
    with open(json_file_name, 'r') as f:
        res = json.load(f)
    for value in res['GraphImages']:
        yield value


if __name__ == '__main__':
    result = {'GraphImages': list()}
    for file_to_add in tqdm(aggr(sys.argv[1])):
        result['GraphImages'].extend(get_contents(file_to_add))

    with open('text_data/all_jsons_old.json', 'w') as f:
        json.dump(result, f)
