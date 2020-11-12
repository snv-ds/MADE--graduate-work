import json
import sys
import re
from tqdm import tqdm


def delete_files(duplicates, json_data):
    for duplicate in tqdm(duplicates, desc='deleting duplicates'):
        for post in json_data['GraphImages']:
            found = re.findall(duplicate.strip(), post['display_url'])
            if found:
                json_data['GraphImages'].remove(post)
                break


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        bad_images = f.readlines()

    with open(sys.argv[2], 'r') as f:
        json_data = json.load(f)

    print(len(json_data['GraphImages']))

    delete_files(bad_images, json_data)

    print(len(json_data['GraphImages']))

    with open('../text_data/all_jsons.json', 'w') as f:
        json.dump(json_data, f)
