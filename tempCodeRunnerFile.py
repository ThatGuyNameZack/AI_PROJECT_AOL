from flask import Flask, render_template
from main import run_emotion_detection  # Import the emotion detection function
from visualization import plot_emotion_confidences  # Import the plotting function

app = Flask(__name__)

@app.route('/')
def index():
    # This will render the index.html page (if you want a homepage)
    return render_template('index.html')

@app.route('/emotion_summary')
def emotion_summary():
    
    # This will process and render the emotion detection summary page
    emotion_avg_confidences = run_emotion_detection()  # Get emotion confidences
    plot_img = plot_emotion_confidences(emotion_avg_confidences)  # Get the base64-encoded plot
    return render_template('emotion_summary.html', confidences=emotion_avg_confidences, plot_img=plot_img)

if __name__ == "__main__":
    app.run(debug=True)
