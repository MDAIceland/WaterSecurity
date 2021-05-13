# %%
import pandas as pd
import sys
sys.path.append("..")
from data.unlabeled.preprocessed import econ, aqua, edu, humdev

# %%
print("Aqua shape:",aqua.shape,"Econ shape:",econ.shape,"Edu shape:",edu.shape,"Humdev shape:",humdev.shape)
# %%
countries_diff_edu_humdev= edu.loc[set(edu.index) - set(humdev.index)]['Short Name']
countries_diff_aqua_humd = aqua.loc[set(aqua.index) - set(humdev.index)]['Country']
countries_not_in_humdev = countries_diff_edu_humdev.append(countries_diff_aqua_humd)
# %%
print("Extra not needed indicators in edu dataset",countries_not_in_humdev)
# %%
countries_diff_humdev_econ = set(humdev.index) - set(econ.index) 
print("Not in humdev",countries_diff_humdev_econ)

# %%
big_table = humdev.join(edu, how="inner").join(aqua, how="inner")
# %%
big_table.dropna(axis=1, inplace=True)


# %%
name_columns = set(big_table.columns) - set(big_table.select_dtypes(include="number").columns)
print("Info columns",name_columns)
# %%
big_table.drop(['Short Name','Long Name','Table Name'],inplace=True, axis=1)

# %%
big_table
# %%
big_table.to_csv("../data/unlabeled/preprocessed_unlabeled.csv")
# %%
