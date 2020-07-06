import json
from collections import defaultdict
from db import Database
from setting import setting
from collections import defaultdict
from tqdm import tqdm

db = Database('movie', setting['host'], 27017, setting['username'], setting['password'])
db2 = Database('movie')


def intersect(list1, list2):
    ''' 求两个列表的交集 '''
    return list(set(list1) & set(list2))


def wrap(data, candidates):
    '''
    在douban的记录上进行添改
    douban和其他来源candidates的"sourceId", "rating", "rateNum", "url", "cover"放入source中
    douban原本的source、sourceId就不要了
    type合并，'nameFrn', 'summary', 'directors', 'country', 'language', 'stars'如无，使用其他来源的
    '''
    # if len(candidates) > 2:
    #     print(candidates)
    srcs = ["douban", "mtime", "maoyan"]
    for key in ["types"]:
        if not data.get(key):
            data[key] = []
        tmp = set(data[key])
        for item in candidates:
            if item.get(key):
                tmp |= set(item[key])
        data[key] = list(tmp)
    for key in ['nameFrn', 'summary', 'directors', 'country', 'language', 'stars']:
        if not data.get(key):
            for item in candidates:
                if item.get(key):
                    data[key] = item[key]
                    break
    key = 'year'
    if not data.get(key):
        for item in candidates:
            if item.get(key):
                data[key] = item[key][:4]
                break
    tmp = dict()
    for item in [data] + candidates:
        tmp[item["source"]] = dict()
        for key in ["sourceId", "rating", "rateNum", "url", "cover"]:
            if item.get(key):
                tmp[item["source"]][key] = item[key]
            if key in data:
                del data[key]
    data['source'] = tmp
    del douban['_id']   # 让_id重新自动生成一个
    db.insert_one('movie', douban)


def same(doc1, doc2):
    '''
    判断两个记录是否指同一个电影，不同的字段可靠性也不同，有先后区别，仅当两者都有指定字段再判断
    首先，name不相同，false；country没有交集，false；year不相同，false
    director, writers, stars依此判断是否有交集，有，true，无，false
    nameFrn相同，true
    '''
    if doc1['name'] != doc2['name']:
        return False
    for key in ['country']:
        if key in doc1 and key in doc2 and not intersect(doc1[key], doc2[key]):
            return False
    for key in ['year']:
        if key in doc1 and key in doc2 and doc1[key] != doc2[key]:
            return False
    for key in ['director', 'writers', 'stars']:
        if key in doc1 and key in doc2:
            if intersect(doc1[key], doc2[key]):
                return True
            else:
                return False
    for key in ['nameFrn']:
        if key in doc1 and key in doc2 and doc1[key] == doc2[key]:
            return True
    return False


cursor = {}
for source in ['douban', 'maoyan', 'mtime']:
    cursor[source] = db.find('details', {"source": source})

# 对name相同的计数，count>1时，_id保留下来
pipeline = [{
        '$group': {
            '_id': "$name",
            'uniqueIds': {
                '$addToSet': '$_id'
            },
            'count': {
                '$sum': 1
            }
        }
    },
    {
        '$match': {
            'count': {
                '$gt': 1
            }
        }
    }
]

for group in tqdm(db.db.details.aggregate(pipeline), total=len(list(db.db.details.aggregate(pipeline)))):
    docs = defaultdict(list)
    for id in group["uniqueIds"]:
        doc = db.find_one('details', id)
        docs[doc["source"]].append(doc)

    if len(docs) == 1:
        continue

    douban_dict = {}
    res = defaultdict(list)
    for douban in docs['douban']:
        douban_dict[douban['sourceId']] = douban
        for mtime in docs['mtime']:
            if same(douban, mtime):
                res[douban['sourceId']].append(mtime)
        size = len(res[douban['sourceId']]) if douban['sourceId'] in res else 0
        if size > 1:
            # 有多于一个的候选可能，判断same有问题，其实same可以改成返回可能性，然后取最大的，
            # 但这种情况很少见，也就五六条，所以不改了；下同
            print(res[douban['sourceId']])
        for maoyan in docs['maoyan']:
            if same(douban, maoyan):
                res[douban['sourceId']].append(maoyan)
        if douban['sourceId'] in res and len(res[douban['sourceId']]) - size > 1:
            print(res[douban['sourceId']])

        if len(res[douban['sourceId']]) > 0:
            wrap(douban_dict[douban['sourceId']], res[douban['sourceId']])

