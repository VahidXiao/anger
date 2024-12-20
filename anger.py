from flask import Flask, request, jsonify
import csv
import io
import re
import base64

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import matplotlib.pyplot as plt

app = Flask(__name__)

# "Anger" word dictionaries for English, Chinese, and Japanese
anger_dictionaries = {
    'en': set(['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed']),
    'zh': set(['生气', '愤怒', '怒火', '暴怒', '恼火']),
    'ja': set(['怒り', 'イライラ', '憤怒', '激怒'])
}

def validate_language(language):
    """Checks if the requested language is supported."""
    return language in anger_dictionaries

def analyze_text(input_text, language):
    """Analyzes the text to detect anger-related words and calculate intensity."""
    anger_words = anger_dictionaries[language]
    # Using a regex to tokenize by word boundaries and convert to lowercase
    words_in_text = re.findall(r'\b\w+\b', input_text.lower())
    matching_words = [word for word in words_in_text if word in anger_words]
    total_words = len(words_in_text)
    intensity = len(matching_words) / total_words if total_words > 0 else 0
    return words_in_text, matching_words, total_words, intensity

@app.route('/detect_anger', methods=['POST'])
def detect_anger():
    """Detects the intensity of anger in a given text."""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    input_text = data.get('text', '')
    if not input_text:
        return jsonify({'error': 'The "text" field is required and cannot be empty'}), 400

    language = data.get('language', 'en').lower()
    if not validate_language(language):
        return jsonify({
            'error': f"Unsupported language: {language}. Supported languages are {list(anger_dictionaries.keys())}"
        }), 400

    try:
        confidence_threshold = float(data.get('confidence_threshold', 0))
        if not 0 <= confidence_threshold <= 1:
            raise ValueError()
    except ValueError:
        return jsonify({'error': 'The "confidence_threshold" must be a number between 0 and 1'}), 400

    words_in_text, matching_words, total_words, intensity = analyze_text(input_text, language)
    anger_detected = intensity >= confidence_threshold

    return jsonify({
        'emotion': 'anger',
        'language': language,
        'input_text': input_text,
        'total_words': total_words,
        'matching_words': len(matching_words),
        'matching_word_list': matching_words,
        'intensity': round(intensity, 4),
        'confidence_threshold': confidence_threshold,
        'anger_detected': anger_detected
    })

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generates a CSV report for anger detection analysis and returns a base64 encoded chart."""
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    input_text = data.get('text', '')
    language = data.get('language', 'en').lower()

    if not validate_language(language):
        return jsonify({
            'error': f"Unsupported language: {language}. Supported languages are {list(anger_dictionaries.keys())}"
        }), 400

    words_in_text, matching_words, total_words, intensity = analyze_text(input_text, language)
    anger_detected = 'Yes' if intensity > 0 else 'No'

    # Create a CSV report
    output = io.StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerow(['Input Text', 'Language', 'Total Words', 'Matching Words', 'Intensity', 'Anger Detected'])
    csv_writer.writerow([input_text, language, total_words, len(matching_words), round(intensity, 4), anger_detected])

    csv_data = output.getvalue()

    # Generate bar chart
    plt.figure(figsize=(5, 2))
    plt.bar(['Total Words', 'Matching Words'], [total_words, len(matching_words)], color=['blue', 'red'])
    plt.title('Anger Detection Analysis')
    plt.xlabel('Metrics')
    plt.ylabel('Count')

    # Convert plot to base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    base64_img = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return jsonify({
        'csv_report': csv_data,
        'chart_image': base64_img
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)