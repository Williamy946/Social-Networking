import pandas as pd
import matplotlib.pyplot as plt
import copy

group_max = 23
topic_max = 41
event_max = 346
thres = 0.3
a = 1
b = 1
c = 1

uu_df_cat = None
for i in range(24):
    uu_df = pd.read_csv('social/social_connections{}.csv'.format(i), sep=',')
    uu_df['sum'] = uu_df['group']/group_max * a + uu_df['topic']/topic_max * b + uu_df['event']/event_max * c
    uu_df = uu_df[uu_df['sum'] >= thres].drop(['group', 'topic', 'event'], axis=1)
    if i == 0:
        uu_df_cat = copy.deepcopy(uu_df)
    else:
        uu_df_cat = pd.concat([uu_df_cat, uu_df], axis=0, ignore_index=True)
uu_df_cat.to_csv('social/social_connections_concat_filtered.csv', index=None)
