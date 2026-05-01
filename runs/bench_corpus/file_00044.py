import tweepy
import pandas as pd
import time
import datetime as dt  	
import os


def get_data(company,time_period,keywords):
    '''
        PARAMS:
            company: STR
            time_period: LIST
            keywords: LIST
            
        Returns: N/A
    '''
    for i in range(len(time_period)):
        
        if i < len(time_period) - 1:
            index = 0

            for kw in keywords:
                filename = company + "_" + time_period[i] + "_" + str(kw.replace(" ","-"))

                os.system("""snscrape --max-result 100 --jsonl twitter-search \
                            '{} since:{} until:{}' > {}.json""".format(kw,time_period[i],time_period[i+1],filename))
                index += 1
                
    return "Successfully Ingested"


if __name__ == '__main__':

    start_time = time.time()

    print("Twitter Data is being ingested")

    # Define the time period
    sd=dt.datetime(2019, 1, 1)
    ed=dt.datetime(2019, 2, 1)
    dates = pd.date_range(sd, ed) 
    time_period = dates.strftime('%Y-%m-%d').tolist()

    # Company list and Keywords 
    companies = ["APPLE", "TESLA", "NETFLIX", "DISNEY"]
    apple_keywords = ["AAPL", "Tim Cook", "Iphone", "Ipad", "Mac", "Apple Watch", "Apple Event"]
    tesla_keywords = ["TSLA","Tesla", "Elon Musk", "Tesla Model X", "Tesla Model S", "Tesla Model Y", "Tesla Model 3"]
    netflix_keywords = ["NFLX", "Netflix", "Reed Hastings", "Stranger Things", "Black Mirror"]
    disney_keywords = ["DIS", "Disney", "Bob Chapek", "Disney Land", "Disney +", "Disney Plus"]

    for company in companies:

        if company == "APPLE":
            get_data(company,time_period,apple_keywords)
            
        elif company == "TESLA":
            get_data(company,time_period,tesla_keywords)
            
        elif company == "NETFLIX":
            get_data(company,time_period,netflix_keywords)
            
        elif company == "DISNEY":
            get_data(company,time_period,disney_keywords)

    print("Twitter Data has been successfully ingested")
    print("--- %s seconds ---" % (time.time() - start_time))