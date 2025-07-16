# 📽️ Movie Recommendation System using SVD

## 📌 Overview

This project implements a **Collaborative Filtering-based Movie Recommendation System** using the **Singular Value Decomposition (SVD)** algorithm from the [Surprise](http://surpriselib.com/) library. It uses the **MovieLens 100k** dataset—a classic benchmark dataset comprising 100,000 ratings by 943 users for 1,682 movies. The goal is to predict user preferences and generate personalized top-N movie recommendations.

## 💡 Key Features

* Utilizes **Matrix Factorization via SVD** for collaborative filtering
* Evaluates performance using **RMSE (Root Mean Squared Error)**
* Provides **Top-N movie recommendations per user**
* Fully interactive **Streamlit-based UI** to explore recommendations
* Integrates a Jupyter notebook backend for training, evaluation, and recommendation logic

---

## ⚙️ Workflow

### Step 1: Load and Prepare Dataset

* The built-in `ml-100k` dataset is used from the Surprise library.
* Data is split into training and test sets using `train_test_split()`.

```python
from surprise import Dataset
from surprise.model_selection import train_test_split

# Load dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)
```

### Step 2: Train SVD Model

* SVD (Singular Value Decomposition) algorithm from Surprise is used.
* The model is trained on the training dataset.

```python
from surprise import SVD

model = SVD()
model.fit(trainset)
```

### Step 3: Evaluate Model

* Predictions are generated on the test set.
* RMSE is used to evaluate prediction accuracy.

```python
from surprise import accuracy

predictions = model.test(testset)
print("RMSE:")
accuracy.rmse(predictions)
```

**Sample Output:**

```
RMSE: 0.9351
```

### Step 4: Generate Top-N Recommendations

* For each user, the top N (default = 5) highest-rated predicted movies are selected.

```python
from collections import defaultdict

def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

recommendations = get_top_n(predictions, n=5)
```

### Step 5: Deploy Interactive Streamlit App

The frontend is built using **Streamlit** to allow users to select a User ID and view their top 5 recommendations.

```python
import streamlit as st

st.title("🎬 Movie Recommendation System using SVD")
user_ids = sorted(recommendations.keys())
selected_user = st.selectbox("Choose User ID", user_ids)

if selected_user:
    for movie_id, rating in recommendations[selected_user]:
        st.write(f"📽️ Movie ID: {movie_id}, ⭐ Predicted Rating: {rating:.2f}")
```

---

## 📊 Sample Output

```
Top 5 Recommendations for User 854:
- Movie ID: 357 | Predicted Rating: 4.11
- Movie ID: 50  | Predicted Rating: 4.01
- Movie ID: 511 | Predicted Rating: 4.01
- Movie ID: 514 | Predicted Rating: 3.93
- Movie ID: 269 | Predicted Rating: 3.85
```

---

## 🚀 Future Improvements

* Incorporate **content-based filtering** for hybrid recommendation
* Add **movie titles and posters** using external movie metadata
* Support **batch recommendation uploads** (CSV input)
* Integrate **user feedback loop** for improved model tuning
* Deploy on **cloud platforms** (Streamlit Cloud, Heroku, AWS)

---

## 🛠️ Technologies Used

* Python 🐍
* Streamlit 🚀
* Surprise (scikit-surprise) 📦
* Jupyter Notebook 📓

---

## 📖 License

This project is open-source and free to use under the **MIT License**.

---

Made with ❤️ using Python, Surprise, and Streamlit.

# Output
![image](https://github.com/user-attachments/assets/39c63a54-f779-4543-9632-fb92f0c0c753)
