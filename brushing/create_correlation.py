import pandas as pd
import numpy as np
import codecs, json


data = []
for i in range(0, 1000):
    element = {}
    temp = []
    p1 = [10, 20]
    p2 = [50, 60]
    p3 = [80, 70]

    temp.append({"t": 0, "out": i})
    temp.append({"t": 1, "out": i})
    temp.append({"t": 2, "out": i})

    element["data"] = temp

    if i < 150:
        x = p1[0] + np.random.randint(-7, 12)
        y = p1[1] + np.random.randint(0, 23)
    elif i >= 150 and i < 300:
        x = p1[0] + np.random.randint(-9, 31)
        y = p1[1] + np.random.randint(0, 26)
    elif i >= 300 and i < 400:
        x = p2[0] - 10 + np.random.randint(-18, 21)
        y = x + np.random.randint(-4, 7)
    elif i >= 400 and i < 600:
        x = p2[0] + np.random.randint(-37, 21)
        y = p2[1] + np.random.randint(-26, 28)
    elif i >= 600 and i < 800:
        x = p2[0] + 10 + np.random.randint(-25, 8)
        y = p2[1] + np.random.randint(-36, 6)
    elif i >= 800:
        x = p3[0] + np.random.randint(-10, 12)
        y = p3[1] + np.random.randint(-12, 15)


    element["params"] = {
        "x": x,
        "y": y
    }

    data.append(element)

data
file_path = "correlation.json"
json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
