import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(f"First 5 rows of the dataset:\n{df.head()}\n")
print(f"Dataset Information:\n{df.info()}\n")
print(f"Missing Values:\n{df.isnull().sum()}\n")
print(f"Statistical Summary:\n{df.describe()}\n"    )
