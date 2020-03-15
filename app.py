from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello world'

# @app.route('/predict', methods=['POST'])
# def predict():
#     return render_template('index.html', image_filename="img/happy.webp", display_mode="none")

# @app.route('/save_pred', methods=['POST'])
# def save_pred():
#     return render_template('index.html', image_filename="img/happy.webp", display_mode="none")


if __name__ == "__main__":
    app.run(debug=True)