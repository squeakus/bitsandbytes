from flask.ext.sendmail import Message

@app.route("/")
def index():

    msg = Message("Hello",
                  sender="from@example.com",
                  recipients=["jonathanbyrn@gmail.com"])
    mail.send(msg)
