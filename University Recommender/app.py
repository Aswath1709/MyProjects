from flask import Flask, render_template, request
import model as m
import recommender as r
app = Flask(__name__, template_folder='template')
@app.route("/")
def home():
    return render_template('home.html')
@app.route("/predictor.html", methods=["GET","POST"])
def model(): 
    return render_template('predictor.html')
@app.route("/result_p", methods=["GET","POST"])
def result_p():
    if request.method=="POST":
        grev=int(request.form["grev"])
        greq=int(request.form["greq"])
        cgpa=float(request.form["cgpa"])
        greawa=float(request.form["greawa"])
        toeflScore=int(request.form["toeflScore"])
        internExp=int(request.form["internExp"])
        confPubs=int(request.form["confPubs"])
        journalPubs=int(request.form["journalPubs"])
        researchExp=int(request.form["researchExp"])
        industryExp=int(request.form["industryExp"])
        program=request.form["program"]
        univ=(request.form["univ"])
        mark=[grev,greq,greawa,cgpa,toeflScore,internExp,confPubs,journalPubs,researchExp,industryExp,program,univ]
        result=m.result_prediction(mark)
        if result==1:
            result="ACCEPTED"
        else:
            result="REJECTED"
        print(mark)
    return render_template('result_p.html',result=result)
@app.route("/recommender.html", methods=["GET","POST"])
def recommender():
    return render_template('recommender.html')
@app.route("/result_r", methods=["GET","POST"])
def result_r():
    if request.method=="POST":
        grev=int(request.form["grev"])
        greq=int(request.form["greq"])
        cgpa=float(request.form["cgpa"])
        greawa=float(request.form["greawa"])
        toeflScore=int(request.form["toeflScore"])
        internExp=int(request.form["internExp"])
        confPubs=int(request.form["confPubs"])
        journalPubs=int(request.form["journalPubs"])
        researchExp=int(request.form["researchExp"])
        industryExp=int(request.form["industryExp"])
        no=int(request.form["no"])
        mark2=[grev,greq,greawa,cgpa,toeflScore,internExp,confPubs,journalPubs,researchExp,industryExp,no]
        rec=r.recommend(mark2)
        print(rec)
    return render_template('result_r.html',result=rec)
if __name__ == "__main__":
    app.run(use_reloader=True)
    app.static_folder = 'static'
