from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS to allow requests from Node.js
import logging
from model import evaluate_code

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for cross-origin requests

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ Health Check Endpoint
@app.route("/", methods=["GET"])
def health_check():
    logging.info("Health check requested")
    return jsonify({"status": "✅ Server is running!", "message": "Flask API is live!"})

# ✅ Evaluation Endpoint
@app.route("/evaluate", methods=["POST"])
def handle_evaluation():
    try:
        # ✅ Ensure JSON Data
        if not request.is_json:
            logging.error("❌ Invalid request: No JSON payload")
            return jsonify({"error": "Invalid request: JSON data required"}), 400

        data = request.get_json()
        logging.info(f"🔹 Received request: {data}")

        # ✅ Validate Required Fields
        required_fields = ["reference_code", "answer_code"]
        for field in required_fields:
            if field not in data:
                logging.error(f"❌ Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # ✅ Evaluate Code
        result = evaluate_code(
            data["reference_code"],
            data["answer_code"],
            data.get("input_data", ""),
            data.get("rubric", {})
        )

        logging.info(f"✅ Evaluation completed. Sending response: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ✅ Start the Flask Server
if __name__ == "__main__":
    logging.info("🚀 Flask App is starting...")
    logging.info("🌍 Running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)  # ✅ Prevent crashes due to threading
