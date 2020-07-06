from db import Database
from setting import setting
import pandas as pd
import re
from tqdm import tqdm
import json
import langid
import opencc
cc = opencc.OpenCC('t2s')


# df = pd.read_excel('豆瓣电影数据集.xlsx', usecols=['电影名称', '豆瓣网址'])
# df['豆瓣网址'] = df['豆瓣网址'].apply(lambda x: re.search('/(\d+)', x.strip('/')).group(1))
# df.index = df['豆瓣网址']
# table = df['电影名称'].to_dict()
# del df


db = Database('movie', setting['host'], 27017, setting['username'], setting['password']).db
# db = Database('movie').db



cnt = 0
# with open('ids.txt', 'w') as f:
#     for item in db['details'].find({'source': 'douban'}, no_cursor_timeout=True):
#         if item.get('nameFrn') and item['source'] == 'douban':
#             if item['sourceId'] in table:
#                 item['name'] = table[item['sourceId']]
#                 item['nameFrn'] = ''
#                 db['details'].replace_one({'_id': item['_id']}, item)
            # else:
            #     if len(item['nameFrn']) < len(item['name']):
            #         item['name'] += item['nameFrn']
            #         item['nameFrn'] = ''
            #         print(item['name'], '|____|', item['nameFrn'])
            #     else:
            #         if item['name'].endswith('季'):
            #             continue
            #         tokens = item['name'].split(' ')
            #         if len(tokens) > 3:
            #             # continue
            #         # else:
            #             print(item['name'], '|____|', item['nameFrn'])
            #             # flag = True
            #             # for c in tokens[1]:
            #             #     if c in tokens[0]:
            #             #         flag = False
            #             #         item['name'] = tokens[0]
            #             #         item['nameFrn'] = tokens[1] + item['nameFrn']
            #             #         break
print(cnt)


# for item in db['details'].find({'source': 'douban', "name": {"$regex": "[\u4e00-\u9fa5].* .*[^\u4e00-\u9fa5\d]"}}, no_cursor_timeout=True):
    # item2 = db['movie'].find_one({'source': 'douban', 'id': item['sourceId']})
    # if item2:
    #     # print(item2['name'], end=' ')
    #     if len(item['name']) > len(item2['name']):
    #         item['nameFrn'] = item['name'][len(item2['name']) + 1:]
    #         item['name'] = item2['name']
    #         # print(item['name'], item['nameFrn'])
    #         db['details'].replace_one({'_id': item['_id']}, item)
    # elif "nameFrn" not in item or item["nameFrn"] == "":
    #     tokens = item['name'].split(' ')
    #     if len(tokens) == 2:
    #         # print(tokens)
    #         item['name'] = tokens[0]
    #         item['nameFrn'] = tokens[1]
    #         db['details'].replace_one({'_id': item['_id']}, item)

def common_char(string1, string2):
    for c in string1:
        if c in string2:
            return True
    return False

cnt = 0
# for item in db['details'].find({'source': 'douban', "name": {"$regex": "[\u4e00-\u9fa5].* .*[^\u4e00-\u9fa5\d]"}},
#                                no_cursor_timeout=True):
#     tokens = item['name'].split(' ')
#     if re.search('[\u3040-\u31FF\uAC00-\uD7AF\u1100-\u11FF]', tokens[0]):
#         continue
#     if len(tokens) == 2 and re.search('[\u3040-\u31FF\uAC00-\uD7AF\u1100-\u11FF]', tokens[1]):
#         item['name'] = tokens[0]
#         item['nameFrn'] = tokens[1]
#         cnt += 1
#         db['details'].replace_one({'_id': item['_id']}, item)
#     # 日文
#     elif re.search('[\u3040-\u31FF]', item['name']):
#         cnt += 1
#         for i in range(1, len(tokens)):
#             if tokens[i] == '':
#                 continue
#             potential = cc.convert(tokens[i])
#             if common_char(potential[:-1], tokens[0][:-1]):
#                 item['name'] = ' '.join(tokens[:i])
#                 item['nameFrn'] = ' '.join(tokens[i:])
#                 # print(item['name'], '|----|', item['nameFrn'])
#                 db['details'].replace_one({'_id': item['_id']}, item)
#                 break
# print(cnt)


digit_dict = {"零": 0, "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3,
             "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, '十': 1}
def convert_arab(zh):
    res = ""
    for c in zh:
        if c in digit_dict:
            res += str(digit_dict[c])
        else:
            res += c
    return res

for item in db['details'].find({"nameFrn": {"$regex": "^[\d-]+$"}},
                               no_cursor_timeout=True):
    if item['name'].startswith(item['nameFrn']):
        continue
    if ' ' in item['name']:
        item['name'] += item['nameFrn']
        item['nameFrn'] = ''
        tokens = item['name'].split(' ')
        for i in range(1, len(tokens)):
            if tokens[i] == '':
                continue
            potential = cc.convert(tokens[i])
            if common_char(potential[:-1], tokens[0][:-1]):
                item['name'] = ' '.join(tokens[:i])
                item['nameFrn'] = ' '.join(tokens[i:])
                print(item['name'], '|----|', item['nameFrn'])
                db['details'].replace_one({'_id': item['_id']}, item)
                break
    elif len(item['nameFrn']) == 1:
        if len(item['name']) == 1:
            arab = convert_arab(item['name'])
            if arab == item['nameFrn']:
                continue
        item['name'] += item['nameFrn']
        item['nameFrn'] = ''
        print(item['name'], '|----|', item['nameFrn'])
        db['details'].replace_one({'_id': item['_id']}, item)
    else:
        arab = convert_arab(item['name'])
        if not item['nameFrn'] in arab:
            item['name'] += item['nameFrn']
            item['nameFrn'] = ''
            print(item['name'], '|----|', item['nameFrn'])
            db['details'].replace_one({'_id': item['_id']}, item)