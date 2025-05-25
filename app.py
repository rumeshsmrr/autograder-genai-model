from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from model import CodeEvaluator  # 🔁 Update your model filename here

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for cross-origin access

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO)

# ✅ Initialize Code Evaluator
evaluator = CodeEvaluator()

# ✅ Health Check
@app.route("/", methods=["GET"])
def health_check():
    logging.info("Health check ping received.")
    return jsonify({
        "status": "✅ Server is running!",
        "message": "Flask API is live!"
    })

# ✅ Evaluate Code
@app.route("/evaluate", methods=["POST"])
def evaluate_submission_route():
    try:
        if not request.is_json:
            logging.error("❌ Request not in JSON format.")
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        logging.info(f"🔹 Received evaluation request: {data}")

        # ✅ Required fields
        required_fields = ["reference_code", "answer_code", "rubric"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logging.error(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 400

        ref_code = data["reference_code"]
        answer_code = data["answer_code"]
        rubric = data["rubric"]
        input_data = data.get("input_data", "")  # Optional input

        if not isinstance(rubric, dict) or not all(isinstance(v, (int, float)) for v in rubric.values()):
            logging.error("❌ Rubric must be a dictionary with numeric values.")
            return jsonify({"error": "Rubric must be a dictionary with numeric values"}), 400

        # ✅ Evaluate Code
        result = evaluator.evaluate_submission(ref_code, answer_code, rubric, input_data)

        response_payload = {
            "status": "success",
            "final_score": result["total_score"],
            "grades": result["grades"],
            "feedback": result["feedback"],
            "code_similarity_percentage": result["code_similarity_percentage"],
            "code_similarity_details": result["code_similarity_details"],
            "syntax_errors": result.get("syntax_errors", [])
        }

        logging.info(f"✅ Evaluation completed. Response: {response_payload}")
        return jsonify(response_payload)

    except Exception as e:
        logging.error(f"❌ Internal server error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ✅ Run the Server
if __name__ == "__main__":
    logging.info("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
