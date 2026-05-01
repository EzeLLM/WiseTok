import requests
import json
import os

"""
The script uses the NX CLI API (not REST).
It asks the user to input a list of VLAN ID's to be deleted.
"""

switchuser=os.environ.get("NXAPI_USER")
switchpassword=os.environ.get("NXAPI_PASS")

url='http://10.10.20.58/ins'
myheaders={'content-type':'application/json'}

user_input = input("Enter the VLANs list (ex. 2,3,4): ")
vlans_list = list(user_input.split(","))

vlans = ""
for vlan in vlans_list:
    vlans += "no vlan "+str(vlan)+" ;"

payload={
  "ins_api": {
    "version": "1.0",
    "type": "cli_conf",
    "chunk": "0",
    "sid": "sid",
    "input": f"{vlans}",
    "output_format": "json"
  }
}

url='https://10.10.20.58/ins'
response = requests.request("POST", url,
    data=json.dumps(payload), 
    auth=(switchuser,switchpassword),
    verify=False).json()

print("+++++   VLAN(s) were deleted.   +++++")