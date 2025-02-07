from flask import Flask, request, jsonify
from src.evaluation.evaluate_mmlu import evaluate_mmlu

app = Flask(__name__)

@app.route('/evaluate_mmlu', methods=['POST'])
def evaluate_mmlu_api():
    """
    API endpoint to evaluate a model on the MMLU dataset.
    Accepts model_name and device in the request body as JSON.
    Returns the accuracy as JSON response.
    """
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        device = data.get('device', 'cpu') # Default to CPU if device is not provided

        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400

        accuracy = evaluate_mmlu(model_name, device)
        return jsonify({'accuracy': accuracy})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
