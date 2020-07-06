import pandas as pd
import pymongo
from setting import setting

db = pymongo.MongoClient(host=setting['host'])['movie']  # 连接数据库
db.authenticate(setting['username'], setting['password'])  # 用户验证
excludes = ['insertStamp', 'timestamp', 'rating', 'rateNum', 'cover']
excludes = {v: False for v in excludes}
data = db['details'].find({}, excludes)
df = pd.DataFrame.from_records(data)
df.to_csv('details.csv', index=False)


doubanIds = []
table = {'maoyan': {}, 'mtime': {}}     # otherSrcId -> doubanId

# 只需要source字段的数据
for doc in db['movie'].find({}, {'source': True}, no_cursor_timeout=True):
    doubanId = doc['source']['douban']['sourceId']
    doubanIds.append(doubanId)
    for src in ['maoyan', 'mtime']:
        if src in doc['source']:
            table[src][doc['source'][src]['sourceId']] = doubanId
print('douban num: ', len(doubanIds))

details = pd.read_csv('details.csv')
douban = details[(details['source'] == 'douban') & details['sourceId'].isin(doubanIds)]
maoyan = details[(details['source'] == 'maoyan') & details['sourceId'].isin(table['maoyan'].keys())]
mtime = details[(details['source'] == 'mtime') & details['sourceId'].isin(table['mtime'].keys())]

df = douban.append(maoyan).append(mtime)  # 拼接起来
trans = {'douban': {}, 'maoyan': {}, 'mtime': {}}
for id, row in df.iterrows():
    trans[row['source']][str(row['sourceId'])] = row['_id']     # srcId -> newId

with open('link.csv', 'w') as f:
    for src in ['maoyan', 'mtime']:
        for k, v in table[src].items():
            f.write("{}\t{}\n".format(trans[src][k], trans['douban'][v]))   # other_id, douban_id

# df['id'] = df.index
details.drop(columns=['imdb', 'sourceId']).to_csv('data.csv', index=False) # drop去掉不需要的列

# cnt = 0
# for item in db['movie'].find({}, no_cursor_timeout=True):
#     cnt += 1
#     print(cnt)
#     db['movie'].update_one({'_id': item['_id']},
#                            {'$set': {'id': trans['douban'][item['source']['douban']['sourceId']]}})







