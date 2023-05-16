const chai = require('chai');
const server = require("../index.js");
const chaiHttp = require('chai-http');
const expect = chai.expect;
chai.should();
chai.use(chaiHttp);

describe('Guest Lecture', () => {


    describe('Route Index', () => {
        it('Should Render Login Page', (done) => {
            chai.request(server)
                .get('/')
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    
                    done();
                });
        });
    });

    describe("/login", () => {
        it("Should Login ", (done) => {
            chai.request(server)
                .get("/students/login?uname=Abishek&pass=Vasanthan")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/login", () => {
        it("Should Login ", (done) => {
            chai.request(server)
                .get("/faculties/login?uname=sabarish&pass=sab12")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/student/added", () => {
        it("Student should be added ", (done) => {
            chai.request(server)
                .get("/students/add?id=CSE18213&name=Aswanth&username=ash&password=rag&elective=15ENG111&department=CSE&section=C&semester=6")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/faculty/added", () => {
        it("Faculty should be added ", (done) => {
            chai.request(server)
                .get("/faculties/add?id=f06&name=Ananth&username=anan&password=anan12&department=CSE")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/course/added", () => {
        it("Course should be added ", (done) => {
            chai.request(server)
                .get("/courses/add?id=15CSE213&name=EmbeddedSystems&semester=4&type=core")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/student/modify", () => {
        it("Student should be modify ", (done) => {
            chai.request(server)
                .get("/students/modify?id=CSE18213&name=Aswanth&username=ash&password=rag&elective=15ENG111&department=CSE&section=C&semester=6")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/faculty/modify", () => {
        it("Faculty should be modified ", (done) => {
            chai.request(server)
                .get("/faculties/modify?id=f06&name=Ananth&username=anan&password=anan12&department=CSE")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/course/modify", () => {
        it("Course should be modified ", (done) => {
            chai.request(server)
                .get("/courses/modify?id=15CSE213&name=EmbeddedSystems&type=core&sem=4")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/student/delete", () => {
        it("Student should be deleted ", (done) => {
            chai.request(server)
                .get("/students/delete?id=CSE18213")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/faculty/delete", () => {
        it("Faculty should be deleted ", (done) => {
            chai.request(server)
                .get("/faculties/delete?id=f06")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/course/delete", () => {
        it("Course should be deleted ", (done) => {
            chai.request(server)
                .get("/courses/delete?id=15CSE213")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/student/login", () => {
        it("Student loggedin ", (done) => {
            chai.request(server)
                .get("/students/login?uname=Abishek&pass=vasanthan")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/faculty/login", () => {
        it("Faculty loggedin ", (done) => {
            chai.request(server)
                .get("/faculties/login?uname=sabarish&pass=sab12")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/faculty/retrieve", () => {
        it("Faculty retrieves ", (done) => {
            chai.request(server)
                .get("/faculties/retrieve?fid=f01&cid=15CSE212&sem=6&sec=C")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });

    describe("/student/viewattendance", () => {
        it("Student view attendance ", (done) => {
            chai.request(server)
                .get("/students/viewattendance?sid=CSE18203&sem=6")
                .end((err, res) => {
                    expect(res).to.have.status(200);
                    done();
                });
        });
    });
});