import json
lable="{},{}".format(28,"F")
print(lable)
dic={"a":lable.split(",")[0],"b":lable.split(",")[1]}
print(dic)
print(json.dumps(dic))
print(lable.split(",")[0],lable.split(",")[1])