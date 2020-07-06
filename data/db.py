# mongodb database
from pymongo import MongoClient


class Database(object):
    def __init__(self, database, address='127.0.0.1', port=27017, name=None, pwd=None):
        self.conn = MongoClient(host=address, port=port)
        self.db = self.conn[database]
        if name:
            self.db.authenticate(name, pwd)

    def insert_one(self, collection, data):
        ret = self.db[collection].insert_one(data)
        return ret.inserted_id

    def insert_many(self, collection, data):
        ret = self.db[collection].insert_many(data)
        return ret.inserted_ids

    def update(self, collection, data):
        # data format:
        # {key:[old_data,new_data]}
        data_filter = {}
        data_revised = {}
        for key in data.keys():
            data_filter[key] = data[key][0]
            data_revised[key] = data[key][1]
        return self.db[collection].update_many(data_filter, {"$set": data_revised}).modified_count

    def find(self, col, condition, column=None):
        if column is None:
            return self.db[col].find(condition)
        else:
            return self.db[col].find(condition, column)

    def find_one(self, col, filter=None, condition=None, column=None):
        return self.db[col].find_one(filter, condition)

    def delete(self, col, condition):
        return self.db[col].delete_many(filter=condition).deleted_count
