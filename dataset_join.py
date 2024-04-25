import numpy as np
import pandas as pd
import os
from PIL import Image
import json
import csv

json_path_dev_seen = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/dev_seen.jsonl'
json_path_dev_unseen = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/dev_unseen.jsonl'
json_path_train = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/train.jsonl'
json_path_test_seen = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/test_seen.jsonl'
json_path_test_unseen = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/test_unseen.jsonl'

json_path = '/Users/kurylo/Desktop/AI/HateSpeech/Dataset/dev.jsonl'
json_path_train = '/Users/kurylo/Desktop/AI/HateSpeech/Dataset/train.jsonl'
json_path_test = '/Users/kurylo/Desktop/AI/HateSpeech/Dataset/test.jsonl'

text_data = []

csv_path = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/HateSpeech.csv'


def save_text_data():
    with open(csv_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'number', 'text', 'label'])
        i = 0
        with open(json_path_dev_seen, 'r') as jsonl_file:
            for line in jsonl_file:
                i += 1
                data = json.loads(line)
                csv_writer.writerow([data['id'], i, data['text'], data['label']])
                text_data.append(data)
        with open(json_path_dev_unseen, 'r') as jsonl_file:
            for line in jsonl_file:
                i += 1
                data = json.loads(line)
                csv_writer.writerow([data['id'], i, data['text'], data['label']])
                text_data.append(data)
        with open(json_path_train, 'r') as jsonl_file:
            for line in jsonl_file:
                i += 1
                data = json.loads(line)
                csv_writer.writerow([data['id'], i, data['text'], data['label']])
                text_data.append(data)
        with open(json_path_test_seen, 'r') as jsonl_file:
            for line in jsonl_file:
                i += 1
                data = json.loads(line)
                csv_writer.writerow([data['id'], i, data['text'], data['label']])
                text_data.append(data)
        with open(json_path_test_unseen, 'r') as jsonl_file:
            for line in jsonl_file:
                i += 1
                data = json.loads(line)
                csv_writer.writerow([data['id'], i, data['text'], data['label']])
                text_data.append(data)


def save_image_data(df):
    c = 0
    for i in df['id']:
        print(len(str(i)))
        if len(str(i)) == 4:
            img_path = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/img/0' + str(i) + '.png'
        if len(str(i)) == 5:
            img_path = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/img/' + str(i) + '.png'
        destination_folder = '/Users/kurylo/Desktop/AI/DiplomaHateSpeech/dataset/images/'
        print(img_path)
        with Image.open(img_path) as img:
            img = img.resize((256, 256))

            destination_path = os.path.join(destination_folder, str(c) + '.png')
            c += 1
            img.save(destination_path)

            print("Image has been processed and saved to the destination folder.")


save_text_data()

df = pd.read_csv(csv_path)

save_image_data(df)