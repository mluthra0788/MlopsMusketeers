from flask import Flask, request, jsonify
import pickle
import numpy as np


# initialize the flask app
app = Flask(__name__)


@app.route("/prediction", methods=["POST"])
def prediction():
    with open("random_forest.pkl", "rb") as f:
        clf = pickle.load(f)
    print(clf)
    payload = request.json
    # Convert the list into a NumPy array and reshape it
    image_data = np.random.randint(0, 255, (28 * 28)).tolist()  # Reshape for model
    prediction = clf.predict(image_data)
    print(payload)
    return jsonify({"predicted_value":prediction[0]})



if __name__ == "__main__":
    app.run(port=5003)

