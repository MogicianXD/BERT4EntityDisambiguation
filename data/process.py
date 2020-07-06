import pandas as pd
import numpy as np
import pymongo
from setting import setting

db = pymongo.MongoClient(host=setting['host'])['movie']  # 连接数据库
db.authenticate(setting['username'], setting['password'])  # 用户验证

data = pd.read_csv('link.csv', sep='\t', header=None)
indice = np.random.permutation(len(data))

split_ratio = 0.6
split_ratio2 = 0.8

# data[:int(len(data) * split_ratio)].to_csv('train.csv', index=False)
# data[int(len(data) * split_ratio): int(len(data) * split_ratio2)].to_csv('valid.csv', index=False)
# data[int(len(data) * split_ratio2):].to_csv('test.csv', index=False)

over = set(data[0].tolist() + data[1].values.tolist())
with open('wait.csv', 'w') as f:
    for doc in db.details.find({'source': {'$ne': 'douban'}}, projection={'_id': True}):
        if doc['_id'] in over:
            continue
        f.write(str(doc['_id']) + '\n')
