import pandas as pd
df = pd.read_csv('/data/XinZhi/ODUQ/DOT/data/chengdu/20181101_20181110.csv')[:20000]
df = df.iloc[:, 1:]
df.to_csv('test_0110.csv')
# df = pd.read_csv('test.csv', usecols=lambda column: column != 'Unnamed: 0')
# df.to_csv('test.csv', index=False)