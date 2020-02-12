import json, os, sys

trace = "/home/alfuerst/tensorflow-simulations/lj-box/outputs/output-10-10000-1000-108/2000-timeline.json"

data=None
with open(trace) as f:
    data = json.load(f)

data = data["traceEvents"]

names = set()
for item in data:
    if "name" in item:
        sects = item["name"].split("/")
        for sect in sects:
            names.add(sect)
# print(sorted(names))
# exit()
compute_ops = ["xla", "add", "less", "const", "reduction_indices", "logicalor", "sqrt", "pow", 
            "square", "isnan", "reshape", "greater", "realdiv", "mul", "sum", "select", "sub",
            "or", "where"]

start = sys.maxsize
end = 0

for item in data:
    if "ts" in item:
        if item["ts"] < start:
            start = item["ts"]
        if item["ts"] > end:
            end = item["ts"]

tot = end - start
print("Elapsed time: ", tot)

wall = 0
for item in data:
    if "dur" in item:
        wall += item["dur"]

print("Active time %: ", (wall/tot)*100)

compute_wall = 0
for item in data:
    for op in compute_ops:
        if op in item["name"].lower():
            if "dur" in item:
                compute_wall  += item["dur"]
print("Productive time %: ", (compute_wall/tot)*100)

where_wall = 0
for item in data:
        if "where" in item["name"].lower()or "select" in item["name"].lower():
            # print(item)
            if "dur" in item:
                where_wall  += item["dur"]
print("Where time %: ", (where_wall/tot)*100)

control_wall = 0
for item in data:
    for op in compute_ops:
        not_in = True
        if op in item["name"].lower():
            not_in = False
            break
    if not_in:
        if "dur" in item:
            control_wall  += item["dur"]
print("Control Flow time %: ", (control_wall/tot)*100)

def calc_lost_time(data):
    op_times = []
    for item in data:
        if "dur" in item:
            op_times.append((item["ts"], item["dur"]))
    op_times = sorted(op_times, key=lambda x: x[0], reverse=False)
    lost_time = 0
    while len(op_times) > 1:
        start, duration = op_times.pop(0)
        if op_times[0][0] <= start + duration + 1:
            continue
        else:
            lost_time += op_times[0][0] - (start + duration+1)
    return lost_time
    
print("Lost time %: ", (calc_lost_time(data)/tot)*100)