from flask import Flask, request, jsonify
from model import evaluate_code

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def handle_evaluation():
    try:
        data = request.get_json()
        print("🔹 Received request:", data)  # Print the incoming request data

        result = evaluate_code(
            data["reference_code"],
            data["answer_code"],
            data.get("input_data", ""),
            data.get("rubric", {})
        )

        print("✅ Evaluation completed. Sending response:", result)  # Log response
        return jsonify(result)
    except KeyError as e:
        error_msg = f"❌ Missing required field: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    print("🚀 Flask App is starting...")  # Log when the app starts
    print("🌍 Running on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
