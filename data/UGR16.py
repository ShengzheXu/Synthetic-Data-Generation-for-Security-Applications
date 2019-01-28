import pandas as pd
import numpy as np

datafile = 'D:\\research_local_data\\april_week3_csv\\april_week3_csv\\uniq\\april.week3.csv.uniqblacklistremoved'

N = 10000
with open(datafile) as myfile:
    head = [next(myfile) for x in range(N)]
print('\n'.join(head[:5]))

head = head[1:]
theDate = head[0].split(',')[0].split(' ')[0] # date '2016-04-11'
userFilter = {}
filteredList = []
for line in head:
    elements = line.split(',')
    if (elements[0].split(' ')[0] != theDate):
        continue
    if (elements[2] in userFilter):
        filteredList.append(line)
        userFilter[elements[2]] += 1
    elif (len(userFilter) < 10000):
        userFilter[elements[2]] = 1
        filteredList.append(line)
    else:
        pass

with open('selected_data.csv', 'w') as file:
    file.write(''.join(filteredList))

print("num of users", len(userFilter))
print("num of records", len(filteredList))
bestUsers = sorted( ((v,k) for k,v in userFilter.items()), reverse=True)
print(bestUsers[:20])
