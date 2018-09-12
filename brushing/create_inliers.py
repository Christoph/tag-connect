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

    if i < 200:
        x = p1[0] + np.random.randint(-7, 10)
        y = p1[1] + np.random.randint(-10, 7)
    elif i >= 200 and i < 400:
        x = p2[0] - 10 + np.random.randint(-8, 2)
        y = p2[1] + np.random.randint(-20, 12)
    elif i >= 400 and i < 600:
        x = p2[0] + np.random.randint(-8, 8)
        y = p2[1] + np.random.randint(-20, 12)
    elif i >= 600 and i < 800:
        x = p2[0] + 10 + np.random.randint(-2, 10)
        y = p2[1] + np.random.randint(-20, 12)
    elif i >= 800:
        x = p3[0] + np.random.randint(-5, 5)
        y = p3[1] + np.random.randint(-15, 15)


    if x >= 40 and x < 50 and y < 60 and y > 50:
        x += int(2 * np.abs(x - 50))
    elif x > 50 and x <= 60 and y < 60 and y > 50:
        x -= int(2 * np.abs(x - 50))


    element["params"] = {
        "x": x,
        "y": y
    }

    data.append(element)

data
file_path = "inlier.json"
json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
