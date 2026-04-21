from flask import Flask, request, jsonify
from flask_cors import CORS
from signature_engine import calculate_match_score

app = Flask(__name__)
# Enable CORS for the React frontend, allowing uploads
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/verify-signature', methods=['POST'])
def verify_signature():
    try:
        # Check if files are part of the request
        if 'reference' not in request.files or 'verification' not in request.files:
            return jsonify({'error': 'Missing reference or verification file'}), 400
            
        ref_file = request.files['reference']
        ver_file = request.files['verification']
        
        if ref_file.filename == '' or ver_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Process and calculate the match score
        score = calculate_match_score(ref_file, ver_file)
        
        # Determine status based on the agreed 80% threshold
        status = "Verified" if score >= 80.0 else "Declined"
        color = "green" if score >= 80.0 else "red"
        
        return jsonify({
            'success': True,
            'match_score': score,
            'status': status,
            'color': color
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Run the server
    # The React app will run on port 5173 (Vite default), so Flask runs on 5000
    app.run(debug=True, port=5000)
