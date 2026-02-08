# Mystery Data Challenge 🔍

## Dataset Description

Welcome to the Mystery Data Challenge! This dataset contains mysterious features that hold the key to unlocking the correct prediction model.

### Files
- `mystery_data.csv` - The main dataset with 1000 rows and 5 columns

### Columns
- **id**: Unique identifier for each row
- **feature_1**: First numerical feature 
- **feature_2**: Second numerical feature
- **feature_3**: Third numerical feature
- **category**: Categorical variable (A, B, C, D, E)
- **target**: Binary target variable (0 or 1)

### The Challenge

Hidden within this dataset is a clue that will lead you to the correct model file. There are 40 different model implementations in the `models/` folder, but only ONE will reveal the flag when run correctly.

### Notes
- The correct model will output an accuracy score
- Flag format: `flag{accuracy}` where accuracy is the 4-decimal accuracy score. Eg: flag{0.9999}
- Make sure you have scikit-learn installed: `pip install scikit-learn pandas`

Good luck, data detective! 🕵️

---