import os

out_list = os.listdir("./exp3/output/model1/")
# out_list.remove('1620126833.1179667')

grab_mask_list = []
grab_bbox_list = []
snake_mask_list = []
snake_bbox_list = []
model_mask_list = []
model_bbox_list = []

for index, i in enumerate(out_list):
    # if index < 14:
    #     continue
    out_data_list = os.listdir("./exp3/output/model1/" + i + "/")
    f = open("./exp3/output/model1/" + i + "/out.txt")
    f.readline()
    grab_mask = f.readline()
    grab_mask_list.append(float(grab_mask[:-1]))
    f.readline()
    grab_bbox = f.readline()
    grab_bbox_list.append(float(grab_bbox[:-1]))
    f.readline()
    snake_mask = f.readline()
    snake_mask_list.append(float(snake_mask[:-1]))
    f.readline()
    snake_bbox = f.readline()
    snake_bbox_list.append(float(snake_bbox[:-1]))
    f.readline()
    model_mask = f.readline()
    model_mask_list.append(float(model_mask[:-1]))
    f.readline()
    model_bbox = f.readline()
    model_bbox_list.append(float(model_bbox[:-1]))

avg = 0
avg = sum(grab_mask_list) / len(grab_mask_list)
print(avg)

avg = 0
avg += sum(grab_bbox_list) / len(grab_bbox_list)
print(avg)

avg = 0
avg += sum(snake_mask_list) / len(snake_mask_list)
print(avg)

avg = 0
avg += sum(snake_bbox_list) / len(snake_bbox_list)
print(avg)

avg = 0
avg += sum(model_mask_list) / len(model_mask_list)
print(avg)

avg = 0
avg += sum(model_bbox_list) / len(model_bbox_list)
print(avg)
