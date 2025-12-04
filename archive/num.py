import pandas as pd
df = pd.read_csv("username.csv")

#print(df.head().to_string())
#print(df.info())
#print(df.describe())
#print(df.shape)
#print(df.loc[2])
#print(df.columns)
#df2 = df[df["Identifier"]>5000]
#print(df2.head())
#df3 = df.sort_values(by="Identifier",ascending=False)
#print(df3)
df4 = df.drop(columns=["Username","Last name"])
print(df4)
df4.to_csv("Stupid.csv",index=False)
print(len(df))


print(df.iloc[0:3,0:2])

df.loc[len(df)] = ["AJVoci",2005,"Anthony","Voci"]
df.to_csv("username.csv",index=False)
print(df.duplicated())
df = df.drop_duplicates()
df.to_csv("username.csv",index=False)
print(df)

