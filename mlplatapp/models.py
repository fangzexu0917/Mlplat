from django.db import models

# Create your models here.
import pymongo
import json


def SavaData(dataName, dictList, overwrite, host, port, database):
    """
    将字典列表存入mongodb
    :param dataName: 存入的数据名
    :param dictList: 存入的数据
    :param overwrite: 是否覆盖数据库中的数据
    :param host:
    :param port:
    :param database:
    :return:
    """
    client = pymongo.MongoClient(host=host, port=port)  # 用于连接到MongoDB的工具
    db = client.get_database(name=database)  # 获取Database具有给定名称和选项的

    n = 0
    if dataName in db.list_collection_names():  # 获取此数据库中所有集合名称的列表，以下操作对重名文件数据进行修改
        if not overwrite:
            n += 1
            dataName = dataName[:dataName.rfind('.')] + str(n) + dataName[dataName.rfind('.'):]
            while dataName in db.list_collection_names():
                n += 1
                dataName = dataName[:dataName.rfind('.') - 1] + str(n) + dataName[dataName.rfind('.'):]
        else:
            db.drop_collection(dataName)  # 删除收藏集

    coll = db.create_collection(name=dataName)  # 在此数据库中创建一个新的集合

    return dataName, coll.insert_many(dictList)  # 插入可迭代的文档


def UpdateData(number, newData, dataName, host, port, database):
    """
    更新dataName中的数据
    :param number: dataName中待更改的值的‘NO’值，根据这个值找到dataName中需要更改的一项
    :param newData:
    :param dataName:
    :param host:
    :param port:
    :param database:
    :return:
    """
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)

    coll = db.get_collection(dataName)  # 获取Collection具有给定名称和选项的
    return coll.find_one_and_update({'NO': number}, {'$set': newData})  # 查找单个文档并进行更新，返回原始文档或更新后的文档，$操作符是MongoDB中的原子操作


def ReadData(dataName, host, port, database):
    """
    读取数据库中dataName中的所有文档
    :param dataName:
    :param host:
    :param port:
    :param database:
    :return:
    """
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)

    coll = db.get_collection(name=dataName)
    doc_list = [it for it in coll.find(projection={"_id": False})]  # projection，应在结果集中返回的字段名称列表或指定要包括或排除的字段的dict
    # doc_json = json.dumps(doc_list)
    return doc_list


def ReadColl(host, port, database):
    """
    读取数据库中的所有数据集合
    :param host:
    :param port:
    :param database:
    :return:
    """
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)
    return db.list_collection_names()


def DropColl(collection, host, port, database):
    """
    删除数据库中的数据集合
    :param collection:
    :param host:
    :param port:
    :param database:
    :return:
    """
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)
    return db.drop_collection(collection)


def RmDoc(doc_filter, collection, host, port, database):
    """
    删除数据集合中的某个文档
    :param doc_filter: 用于找到需要删除的文档
    :param collection:
    :param host:
    :param port:
    :param database:
    :return:
    """
    client = pymongo.MongoClient(host=host, port=port)
    db = client.get_database(name=database)
    coll = db.get_collection(collection)
    return coll.remove(doc_filter)  # 从此集合中删除文档


class Data(models.Model):
    """
    上传的数据集模型，记录数据集的描述信息，对应的特征数据存储在MongoDB中
    为未来使用关系型数据库准备
    """
    data_name = models.CharField(max_length=200, primary_key=True)
    data_abstract = models.CharField(max_length=200, blank=True, null=True)
    data_keywords = models.CharField(max_length=200, blank=True, null=True)
    data_field = models.CharField(max_length=200, default='')
    data_interest = models.CharField(max_length=200, default='')
    sample_num = models.IntegerField(default=0)
    dim_num = models.IntegerField(default=0)

    submitter = models.CharField(max_length=200, default='')
    collater = models.CharField(max_length=200, default='')
    submitter_organization = models.CharField(max_length=200, blank=True, null=True)
    submitter_email = models.EmailField(default='')
    submitter_phone = models.CharField(max_length=200, blank=True, null=True)
    submitter_address = models.CharField(max_length=200, blank=True, null=True)

    origin = models.CharField(max_length=200, default='')
    origin_type = models.CharField(max_length=200, default='')
    origin_decision = models.CharField(max_length=200, blank=True, null=True)
    origin_platenumber = models.CharField(max_length=200, blank=True, null=True)
    origin_materialname = models.CharField(max_length=200, blank=True, null=True)
    origin_expcondition = models.CharField(max_length=200, blank=True, null=True)
    origin_exparguments = models.CharField(max_length=200, blank=True, null=True)
    origin_expdevice = models.CharField(max_length=200, blank=True, null=True)

    pub_date = models.DateTimeField('date published')


class MachineLearningMethods(models.Model):
    method_name = models.CharField(max_length=200, primary_key=True)
