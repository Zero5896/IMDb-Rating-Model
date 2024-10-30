# 🎬 IMDB Movie Review Sentiment Analysis

<img src="https://th-thumbnailer.cdn-si-edu.com/vCjrNaMJS4XDwEktzN75MD1wdqs=/1000x750/filters:no_upscale()/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/bc/32/bc328d3e-22e2-41e4-b3d5-a230015d7f00/42-36911842.jpg" alt="Cool Porsche 930 Turbo" width="250" height="200"/>


## 🍿 Project Overview
This project focuses on sentiment classification using the IMDB dataset, which includes 50,000 highly polar movie reviews. The goal is to build a model that can accurately classify movie reviews as positive or negative, leveraging both traditional and deep learning approaches.

## 📊 Dataset Summary
The IMDB dataset consists of:
- **50,000 movie reviews** labeled as either positive or negative.
- **Training set**: 25,000 reviews
- **Testing set**: 25,000 reviews

More information about the dataset is available [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data).

## 🎯 Objectives
- **Binary Sentiment Classification**: Predict whether a review is positive or negative.

## 📂 Project Structure
- `data/`: Contains the IMDB dataset.
- `Movie_Review_Logistic_Regression.ipynb`: Model with Logistic Regression
- `Movie_Review_Neural_Networks.ipynb` :Model with Neural Netowrks
- `README.md`: Project documentation.

## 🔍 Methodology
1. **Data Preprocessing**: Steps to clean and prepare text data for modeling.
```python
# Join all reviews into a single string
all_text = ' '.join(df['cleaned_reviews'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Blues', max_words=100).generate(all_text)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![Word Cloud](results/MovieWordCloud.png)

2. **Exploratory Data Analysis (EDA)**: Insights into the distribution and structure of the dataset.
3. **Modeling**:
   - Logistic Regression
```python

```

   - Deep learning models
```python

```

## 📈 Results
- ****

## 📚 Key Learnings
- ****

## 🚀 Future Improvements
- ****
## 📜 References
- [Dataset Source](link_to_dataset_source)
- Additional resources on NLP and sentiment analysis.


## 🤝 Contributing
Feel free to contribute! Please fork the repository and submit a pull request. You can also open an issue if you find a bug or have a suggestion.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.
