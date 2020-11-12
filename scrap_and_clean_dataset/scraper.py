import subprocess
import tqdm
import os


def read_file(file_name):
    with open(file_name, 'r') as f:
        result = f.readlines()
        for tag in result:
            yield tag.strip()


def delete_file_formats(path, file_format='mp4', verbose=True):
    for f in os.listdir(path):
        if f.endswith('.' + file_format):
            os.remove(os.path.join(path, f))
            if verbose:
                print(os.path.join(path, f))


def scrap_tag(tag, max_uploads=1000):
    output = subprocess.run(['python3', 'venv/bin/instagram-scraper', '--tag', tag, '--maximum', str(max_uploads),
                             '--media-metadata', '--media-types', 'image', '-q'])

    delete_file_formats(tag, 'mp4', verbose=False)
    return output


if __name__ == '__main__':
    for tag in read_file('../text_data/top_tags.txt'):
        print(tag)
        scrap_tag(tag)
