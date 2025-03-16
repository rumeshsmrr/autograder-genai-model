from flask import Flask, request, jsonify
from model import evaluate_code

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def handle_evaluation():
    try:
        data = request.get_json()
        result = evaluate_code(
            data["reference_code"],
            data["answer_code"],
            data.get("input_data", ""),
            data.get("rubric", {})
        )
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Using Flask's built-in server for development
    app.run(host='0.0.0.0', port=5000, debug=True)
