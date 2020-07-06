import json

import pandas as pd
import pymongo
from setting import setting
from bson import ObjectId
import re
import threading
import time

db = pymongo.MongoClient(host=setting['host'])['movie']  # 连接数据库
db.authenticate(setting['username'], setting['password'])  # 用户验证

# db = pymongo.MongoClient()['movie']

# df = pd.read_excel('豆瓣电影数据集.xlsx', usecols=['豆瓣网址', '评分', '评价人数', '宣传海报链接'])
# df['豆瓣网址'] = df['豆瓣网址'].apply(lambda x: re.search('/(\d+)/', x).group(1))
# df.columns = ['rating', 'rateNum', 'sourceId', 'cover']
# df = df.drop_duplicates('sourceId')
# df = df.set_index('sourceId')
# table = df.to_dict('index')
# del df

details = pd.read_csv('details.csv')

# past = set([v['_id'] for v in db.details.find({}, projection={'_id': True})])

def recover(data):
    global table
    global db
    # global past
    batch = list()
    for i, row in data.iterrows():
        item = dict()
        # if ObjectId(row['_id']) in past:
        #     continue
        for k, v in row.items():
            if pd.isna(v) or not v:
                continue
            if k == '_id':
                v = ObjectId(v)
            elif k == 'runtime':
                v = int(re.search('(\d+)', str(v)).group(1))
            elif k in ['imdb', 'sourceId', 'year']:
                if v != '未知':
                    v = str(int(v))
                else:
                    v = ''
            elif k in ['country', 'directors', 'language', 'releaseDate', 'stars', 'types', 'writers']:
                v = eval(v)
            item[k] = v
        if item['source'] == 'douban' and item['sourceId'] in table:
            if not pd.isna(item['rating']):
                item['rating'] = table[item['sourceId']]['rating']
            if not pd.isna(item['rateNum']):
                item['rateNum'] = int(table[item['sourceId']]['rateNum'])
            if not pd.isna(table[item['sourceId']]['cover']):
                item['cover'] = table[item['sourceId']]['cover']
        origin = db.profile.find_one({'id': item['sourceId'], 'source': item['source']})
        if origin:
            if origin.get('rate'):
                item['rating'] = origin['rate']
            if origin.get('cover'):
                item['cover'] = origin['cover']
        if 'rating' not in item:
            item['rating'] = ''
        item['timestamp'] = int(time.time())
        print(i)
        batch.append(item)
        if i > 0 and i % 256 == 0:
            db['details'].insert_many(batch)
            batch = []
    if batch:
        db['details'].insert_many(batch)


# start = 138000
# size = 5000
# for p in range(1):
#     print(p)
#     if start >= len(details):
#         break
#     t = threading.Thread(target=recover, args=(details.iloc[start + p * size: start + (p+1) * size], ))
#     t.start()

size = 256
cnt = 0
with open('details.json', encoding='utf-8') as f:
    batch = []
    for line in f:
        doc = json.loads(line.strip())
        doc['_id'] = ObjectId(doc['_id']['$oid'])
        if not doc.get('rateNum'):
            doc['rateNum'] = 0
        if doc['rating'] == 0 and doc['rateNum'] == 0:
            doc['rating'] = ''
        elif type(doc['rating']) != str:
            doc['rating'] = '%.1f' % doc['rating']
        batch.append(doc)
        cnt += 1
        if cnt % size == 0:
            db['details'].insert_many(batch)
            batch = []
        print(cnt)
