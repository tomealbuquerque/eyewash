# import modules
from pymongo import MongoClient

CONNECTIONSTRING = 'mongodb://admin:admin@mongodb:27017'

# connect the Mongo Client
client = MongoClient(CONNECTIONSTRING)

# create the db
db = client.washery_stand