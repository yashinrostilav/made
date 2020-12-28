from classifier import Prophet
from flask import Flask, render_template, request

APPLICATION_NAME = "GenrePredict"

app = Flask(APPLICATION_NAME)

prophet = Prophet()


@app.route("/", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        prediction_message = prophet.predict(text)
    return render_template("page.html", text=text, prediction_message=prediction_message)


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
