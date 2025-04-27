from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors  # <-- NEW

pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))

# Initialize KNN model
model = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')
model.fit(pt)

app = Flask(__name__)

@app.route('/')
def recommend_ui():
    return render_template('recommend.html', data=None, user_input="")

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    user_input_lower = user_input.lower()

    pt_titles_lower = [title.lower() for title in pt.index]

    if user_input_lower in pt_titles_lower:
        index = pt_titles_lower.index(user_input_lower)

        distances, indices = model.kneighbors([pt.iloc[index].values])

        data = []
        for i in indices[0][1:]:  # Skip the first one (it's the book itself)
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)
        
        return render_template('recommend.html', data=data, user_input=user_input)
    
    else:
        return render_template('recommend.html', data=[], user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
