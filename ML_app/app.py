from flask import Flask, render_template, Response, jsonify
from recognizer.camera import VideoCamera

app = Flask(__name__)
camera = VideoCamera()  # Instance de ta caméra + modèle

@app.route('/')
def index():
    return render_template('index.html')  # au lieu d’un string HTML

@app.route('/live')
def live():
    return render_template('live.html')  # Affiche la page avec <img> vers /video_feed

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/alert")
def get_alert():
    # Renvoie l'alerte actuelle au format JSON
    return jsonify({"alert": camera.alert_message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
