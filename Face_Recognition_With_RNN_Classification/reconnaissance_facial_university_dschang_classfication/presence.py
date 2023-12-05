import pandas as pd
import csv



df_state = pd.read_csv(r"presence.csv")



DF_RM_DUP = df_state.drop_duplicates(keep=False)


DF_RM_DUP.to_csv('test.csv', index=False) 