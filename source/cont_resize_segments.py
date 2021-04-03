import pandas as pd
import math


old_len_of_segments = 3
new_len_of_segments = 5
path = '../cont_features/segments_nan/all_feautures.csv'

def roundup(x, to_number):
    return int(math.ceil(x / to_number)) * to_number


df = pd.read_csv(path)
df = df.sort_values(by=['username', 'pattern_id',
                        'segment']).reset_index(drop=True)

new_pattern_id = []
multiplier = -1


for i in range(len(df)):
    if df['segment'][i] == 0:
        multiplier += 1
    new_pattern_id.append((old_len_of_segments * multiplier) + df['segment'][i])


new_new_pattern_id = []
counter = 0
new_new_pattern_id.append(counter)
counter+=1
for i in range(1, len(new_pattern_id)):
    if new_pattern_id[i]-1 != new_pattern_id[i - 1]:
        counter = roundup(counter, new_len_of_segments)
    new_new_pattern_id.append(counter)
    counter += 1

df['pattern_id'] = new_new_pattern_id
df['segment'] = df['pattern_id'] % new_len_of_segments
df['pattern_id'] //= new_len_of_segments

df['id'] = df['id'].astype(str).str.strip().str[-7:]
df['id'] = df['pattern_id'].astype(str) + df['id'].astype(str) 

df.to_csv(path, encoding='utf-8', index=False)
