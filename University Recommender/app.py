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
        grev=int(request.form["greV"])
        greq=int(request.form["greQ"])
        cgpa=float(request.form["cgpa"])
        greawa=float(request.form["greA"])
        program=request.form["program"]
        stream=request.form["stream"]
        univ=(request.form["university"])
        citi=request.form["citizenship"]
        gre_subject=(request.form["gre_subject"])
        if gre_subject=="":
            gre_subject=-1
        else:
            gre_subject=int(gre_subject)
        mark=[univ,stream,program,cgpa,grev,greq,greawa,gre_subject,citi]
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
        grev=int(request.form["greV"])
        greq=int(request.form["greQ"])
        cgpa=float(request.form["cgpa"])
        greawa=float(request.form["greA"])
        program=request.form["program"]
        stream=request.form["stream"]
        univ=(request.form["no"])
        citi=request.form["citizenship"]
        gre_subject=(request.form["gre_subject"])
        if gre_subject=="":
            gre_subject=-1
        else:
            gre_subject=int(gre_subject)
        mark2=[stream,program,cgpa,grev,greq,greawa,gre_subject,citi]
        rec=r.recommend(mark2,int(univ))
        print(rec)
    return render_template('result_r.html',result=rec)
if __name__ == "__main__":
    app.run(use_reloader=True)
    app.static_folder = 'static'
