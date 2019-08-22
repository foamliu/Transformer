import xml.etree.ElementTree

valid_en_old = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en-zh.en.sgm'
valid_en_new = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en'
valid_zh_old = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en-zh.zh.sgm'
valid_zh_new = 'data/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.zh'


def convert(old, new):
    print('old: ' + old)
    print('new: ' + new)
    with open(old, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [line.replace(' & ', ' &amp; ') for line in data]
    with open(new, 'w', encoding='utf-8') as f:
        f.writelines(data)

    root = xml.etree.ElementTree.parse(new).getroot()
    data = [elem.text.strip() + '\n' for elem in root.iter() if elem.tag == 'seg']
    with open(new, 'w', encoding='utf-8') as file:
        file.writelines(data)


if __name__ == '__main__':
    convert(valid_en_old, valid_en_new)
    convert(valid_zh_old, valid_zh_new)
