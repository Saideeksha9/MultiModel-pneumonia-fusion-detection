from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='../templates')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return jsonify({"status": "Success", "message": "Welcome to Python on Vercel!"})

# Required for Vercel Serverless Function handler
if __name__ == '__main__':
    app.run()