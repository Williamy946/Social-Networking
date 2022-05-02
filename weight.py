import pandas as pd
import numpy as np
import time


# 用于对输出任务进度中的时间进行格式化
def fmt_time(dtime):
    if dtime <= 0:
        return '0:00.000'
    elif dtime < 60:
        return '0:%02d.%03d' % (int(dtime), int(dtime * 1000) % 1000)
    elif dtime < 3600:
        return '%d:%02d.%03d' % (int(dtime / 60), int(dtime) % 60, int(dtime * 1000) % 1000)
    else:
        return '%d:%02d:%02d.%03d' % (int(dtime / 3600), int((dtime % 3600) / 60), int(dtime) % 60,
                                      int(dtime * 1000) % 1000)


def count_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    inter_set = set1.intersection(set2)
    return len(inter_set)


group_num = 482
group_weight = 0.5
topic_num = 500
topic_weight = 0.5
event_num = 20
event_weight = 0.8

u_df = pd.read_csv('user_features.csv', sep='\t')
user_num = u_df.shape[0]

inter_group_max = 0
inter_topic_max = 0
inter_event_max = 0

total = int(user_num ** 2 / 2)
count = 0
start = time.time()
with open('social_connections2.csv', mode="w", encoding='utf-8') as out_f:
    line = "uid_i,uid_j,group,topic,event\n"
    out_f.write(line)
    for i in range(user_num):
        ui_group = list(filter(None, u_df.loc[i, 'groups'][1:-1].split(',')))
        ui_topic = list(filter(None, u_df.loc[i, 'topic'][1:-1].split(',')))
        ui_event = list(filter(None, u_df.loc[i, 'event_yes'][1:-1].split(',')))
        ui_group = list(map(int, ui_group))
        ui_topic = list(map(int, ui_topic))
        ui_event = list(map(int, ui_event))
        for j in range(i+1, user_num):
            uj_group = list(filter(None, u_df.loc[j, 'groups'][1:-1].split(',')))
            uj_topic = list(filter(None,  u_df.loc[j, 'topic'][1:-1].split(',')))
            uj_event = list(filter(None, u_df.loc[j, 'event_yes'][1:-1].split(',')))
            uj_group = list(map(int, uj_group))
            uj_topic = list(map(int, uj_topic))
            uj_event = list(map(int, uj_event))
            inter_group = count_same_elem(ui_group, uj_group)
            inter_topic = count_same_elem(ui_topic, uj_topic)
            inter_event = count_same_elem(ui_event, uj_event)
            if inter_group > inter_group_max:
                inter_group_max = inter_group
                print("Max inter_group changed: ", inter_group_max)
            if inter_topic > inter_topic_max:
                inter_topic_max = inter_topic
                print("Max inter_topic changed: ", inter_topic_max)
            if inter_event > inter_event_max:
                inter_event_max = inter_event
                print("Max inter_event changed: ", inter_event_max)

            if not (inter_group == 0 and inter_topic == 0 and inter_event == 0):
                line = "{},{},{},{},{}\n".format(i, j, inter_group, inter_topic, inter_event)
                out_f.write(line)

            count += 1
            if (count % 100000) == 0 or count == total:
                elapsed = time.time() - start
                eta = (total - count) / count * elapsed
                print("[{}/{}], Elapsed: {}, ETA: {}".format(count, total, fmt_time(elapsed), fmt_time(eta)))


print('Max inter_group/inter_topic/inter_event: {}/{}/{}'.format(inter_group_max, inter_topic_max, inter_event_max))
