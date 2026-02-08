import pandas as pd

df = pd.read_csv(r"C:\Users\girik\Desktop\IITK_CR2\datasets\normal.csv")
print("normal.csv")
print(df.shape)
print(df.columns)
print(df.head())
print("\n\n")

df = pd.read_csv(r"C:\Users\girik\Desktop\IITK_CR2\datasets\attack.csv")
print("attack.csv")
print(df.shape)
print(df.columns)
print(df.head())
print("\n\n")

df = pd.read_csv(r"C:\Users\girik\Desktop\IITK_CR2\datasets\merged.csv")
print("merged.csv")
print(df.shape)
print(df.columns)
print(df.head())
print("\n\n")