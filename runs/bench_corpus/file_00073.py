import csv
def write_csv(data):
    headers = data[0].keys()
    with open('C:\Users\Zestl Software\PycharmProjects\Assignments1\outputdata6.csv', 'w') as outfile:
        mywriter = csv.DictWriter(outfile, fieldnames=headers)
        mywriter.writeheader()
        mywriter.writerows(data)
