from pymongo import MongoClient


CONNECTIONSTRING = 'mongodb://admin:{MONGODB_ADMIN}@mongodb:27017'

# connect the Mongo Client
client = MongoClient(CONNECTIONSTRING)

# create the db
db = client.washery_standB