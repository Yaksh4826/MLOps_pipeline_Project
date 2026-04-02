# UCI Obesity Levels Dataset (ID: 544)

## Dataset Overview

This is a **sample version** of the UCI Estimation of Obesity Levels dataset (ID: 544), created with the same structure and features as the original dataset from the UCI Machine Learning Repository.

**Original Dataset Citation:**
Estimation of Obesity Levels Based On Eating Habits and Physical Condition [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5H31Z

---

## Dataset Details

- **Total Records:** 2,111
- **Total Features:** 17 (16 features + 1 target)
- **Missing Values:** None
- **Source Countries:** Mexico, Peru, Colombia
- **Data Collection:** 23% real data, 77% synthetic (SMOTE)

---

## Features Description

### Continuous Features (6):
1. **Age** - Age in years (14-61)
2. **Height** - Height in meters (1.45-1.98)
3. **Weight** - Weight in kilograms (39-173)
4. **FCVC** - Frequency of vegetable consumption (1-3)
5. **NCP** - Number of main meals daily (1-4)
6. **CH2O** - Daily water consumption in liters (1-3)

### Categorical/Binary Features (10):
7. **Gender** - Male/Female → encode as (0/1)
8. **family_history_with_overweight** - Yes/No → encode as (0/1)
9. **FAVC** - Frequent consumption of high caloric food: Yes/No → encode as (0/1)
10. **SMOKE** - Do you smoke? Yes/No → encode as (0/1)
11. **CAEC** - Food consumption between meals: no/Sometimes/Frequently/Always → encode as (0/1/2/3)
12. **SCC** - Calories consumption monitoring: Yes/No → encode as (0/1)
13. **FAF** - Physical activity frequency (0-3)
14. **TUE** - Time using technology devices in hours (0-2)
15. **CALC** - Alcohol consumption: no/Sometimes/Frequently/Always → encode as (0/1/2/3)
16. **MTRANS** - Transportation used: Automobile/Motorbike/Bike/Public_Transportation/Walking → encode as (0/1/2/3/4)


## Binary Classification Version

**File:** `obesity_dataset_binary.csv`

### Binary Target: "Obese" (0/1)

**Conversion Strategy:**
- **Class 0 (Not Obese):** Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II
- **Class 1 (Obese):** Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

**Class Distribution:**
- Not Obese (0): 1,204 samples (57%)
- Obese (1): 907 samples (43%)

**Balance:** Reasonably balanced (57:43 ratio) 

---

## Recommended Feature Encoding

For machine learning models, encode categorical features as follows:

```python
# Binary features: Yes/No
binary_map = {'yes': 1, 'no': 0}

# Gender
gender_map = {'Male': 1, 'Female': 0}

# CAEC (food between meals)
caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

# CALC (alcohol consumption)
calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

# MTRANS (transportation)
mtrans_map = {
    'Walking': 0, 
    'Bike': 1, 
    'Motorbike': 2, 
    'Public_Transportation': 3, 
    'Automobile': 4
}
```

---

## License

Original dataset licensed under Creative Commons Attribution 4.0 International (CC BY 4.0)

## Note

This is a **sample dataset** with synthetic data matching the structure of the original UCI dataset. For the actual dataset with real + SMOTE-generated data, download from:
https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
