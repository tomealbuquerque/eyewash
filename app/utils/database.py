from pymongo import MongoClient


def get_db_connections():

    USER_ENGAGEMENT = 'admin'
    PASSWORD_ENGAGEMENT = 'admin'
    MONGODB_ADMIN = 'admin'

    client_write = MongoClient(f"mongodb://admin:{MONGODB_ADMIN}@mongodb:27017")
    db = client_write.admin

    return db
