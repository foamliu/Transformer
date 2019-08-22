import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    extract('data/ai_challenger_translation_train_20170904.zip')
    extract('data/ai_challenger_translation_validation_20170912.zip')
