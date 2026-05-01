import http.client
import json
import base64
from io import StringIO

USERNAME = 'admin'
PWD = '<PWD>'

BATCH_SIZE = 10000
SCROLL_TIME = "20m"

categoryIndices = ['restored_document_name']

conn = http.client.HTTPConnection("localhost:9200")
csvfile = open("docdata.csv", "a+")

base64string = base64.encodestring("{}:{}".format(USERNAME, PWD))
authheader =  {'Authorization': 'Basic {}'.format(base64string[:-1]),'Content-Type': 'application/json'}

scrollBody = "{}\"scroll\": \"{}\", \"scroll_id\": \"{}\"{}"

def getDocCount(conn, categoryIndex):
    conn.request("GET", "{}/_count".format(categoryIndex), headers = authheader)
    res = conn.getresponse()
    io = StringIO(res.read())
    obj = json.load(io)
    return obj['count']

def writeDocData(resJson):
    for docdata in resJson['hits']['hits']:
        docdata_ = DocData(docdata['_id'], docdata['_source'])
        #docdata.printData()
        csvfile.write(docdata_.getCommaSeparated())

def changeNoneToEmpty(value):
    if value is None:
        return ''
    else:
        return value

class DocData:
        def __init__(self, id, obj):
            self.id = changeNoneToEmpty(id)

        def getCommaSeparated(self):
                return \
                    "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(
                        self.id)

for categoryIndex in categoryIndices:
    totalDocCount = getDocCount(conn, categoryIndex)
    totalRequiredRequests = int(totalDocCount) / BATCH_SIZE
    if int(totalDocCount) % BATCH_SIZE != 0:
        totalRequiredRequests += 1
    print("Nbr of documents in {} : {}, BATCH_SIZE: {}, Nbr of batches: {}".format(categoryIndex, totalDocCount, BATCH_SIZE, totalRequiredRequests))

    conn.request("GET", "/{}/_search?scroll={}&size={}".format(categoryIndex, SCROLL_TIME, BATCH_SIZE), headers = authheader)
    res = conn.getresponse()
    io = StringIO(res.read())
    obj = json.load(io)
    scrollId = obj['_scroll_id']
    writeDocData(obj)

    for i in range(1, totalRequiredRequests, 1):
        print("Key: {}, Running Batch: {} ...".format(categoryIndex, i + 1))
        conn.request("GET", "/_search/scroll", scrollBody.format("{", SCROLL_TIME, scrollId, "}"), headers = authheader)
        res = conn.getresponse()
        io = StringIO(res.read())
        obj = json.load(io)
        writeDocData(obj)

csvfile.close()
