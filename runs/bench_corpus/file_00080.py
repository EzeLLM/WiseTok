'''
Database Operation
'''

from __init__ import shared_mongodb_client

import logging
import traceback


# Get information about all the database topics
def get_db_topics():
    topics = []

    try:
        # Connect to the database
        db = shared_mongodb_client["yousights"]

        # Connect to the collections
        db_collection = db["topics"]

        # Find all the documents
        documents = db_collection.find({"all_matcher": "a"})
    except Exception as error:
        logging.error(f"Error while connecting to the database: {error}")
        logging.error(traceback.format_exc())

    for document in documents:
        this_name = document["name"]
        this_name_in_db = document["name_in_db"]
        this_real_time_query = document["real_time_query"]
        this_topic = {"name": this_name, "name_in_db": this_name_in_db, "real_time_query": this_real_time_query}
        topics.append(this_topic)

    return topics
