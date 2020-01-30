import json, os, sys

trace = "/home/alfuerst/tensorflow-simulations/lj-box/outputs/output-10-10000-1000-108/2000-timeline.json"


data=None
with open(trace) as f:
    data = json.load(f)

data = data["traceEvents"]
# print(len(data))

names = set()
for item in data:
    if "name" in item:
        names.add(item["name"])

# print(len(names))
# for name in names:
#     print(name)
# #print(names)
# exit()
compute_ops = ["xla", "add", "less", "const", "reduction_indices", "logicalor", "sqrt", "pow", "square", "isnan", "reshape", "greater", "realdiv", "mul", "sum", "select", "sub"]

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

print("Productive time %: ", wall/tot)

compute_wall = 0
for item in data:
    for op in compute_ops:
        if op in item["name"].lower():
            if "dur" in item:
                compute_wall  += item["dur"]
print("Compute Productive time %: ", compute_wall/tot)

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
print("Control Flow time %: ", control_wall/tot)

# print((compute_wall+control_wall)/tot)

# with_dur = 0
# op_count = 0
# for item in data:
#     if "cat" in item and item["cat"] != "DataFlow":
#         op_count += 1
# for item in data:
#     if "dur" in item:
#         with_dur += 1

# print(with_dur/op_count)