import replicate
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

@app.route('/enhance', methods=['POST'])
def enhance():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        output = replicate.run(
            "tencentarc/gfpgan",
            input={"img": image_url}
        )
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
