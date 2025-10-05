require("dotenv").config()
const express = require('express');
const cors = require('cors');
const mysql = require('mysql');
const bodyParser = require('body-parser')
const app = express();
app.use(cors());
const nodemailer = require("nodemailer");

app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ limit: '10mb', extended: true }));
const connection = mysql.createConnection({
    host: process.env.DB_HOST,
    user: process.env.DB_USERNAME,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_DATABASE,
})



const SELECT_ALL_STUDENTS_QUERY = 'SELECT * FROM student';
app.get('/', (req, res) => { res.send("Hello from server") });
app.get('/students', (req, res) => {
    connection.query(SELECT_ALL_STUDENTS_QUERY, (err, results) => {
        if (err) {
            return res.send(err)
        }
        else {
            return res.json({
                data: results
            })
        }
    })
})

// const SELECT_ALL_STUDENTS_QUERY = 'SELECT S_Username,S_Password FROM student where id like ${id}';
app.get('/students/auth', (req, res) => {
    var {id} = req.query;
    const SELECTSTUDENT_QUERY = `SELECT S_Username,S_Password FROM student where id like '${id}'`
    connection.query(SELECTSTUDENT_QUERY, (err, results) => {
        if (err) {
            console.log(err)
            return res.send(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculties/auth', (req, res) => {
    var {id} = req.query;
    const SELECTSTUDENT_QUERY = `SELECT F_Username,F_Password FROM faculty where id like '${id}'`
    connection.query(SELECTSTUDENT_QUERY, (err, results) => {
        if (err) {
            console.log(err)
            return res.send(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})



app.post('/msg/post', (req, res) => {
    console.log(req.body)
    doc = (req.body.doc === null ? null : `'${req.body.doc}'`)
    const INSERT_MSG_QUERY = `INSERT INTO msg values (${0},'${req.body.sid}','${req.body.fid}','${req.body.msg}',${0},${doc},'${req.body.date}')`
    // console.log(id, name, username, password, elective, department, section, semester)
    connection.query(INSERT_MSG_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            const SELECT_ALL_FACULTIES_QUERY1 = `SELECT email FROM f_email where id like '${req.body.fid}'`;
            connection.query(SELECT_ALL_FACULTIES_QUERY1, (err, results) => {
                if (err) {
                    // return res.send(err)
                    console.log(err)
                }
                else {
                    console.log(results[0].email)
                    let mailTransporter = nodemailer.createTransport({
                        service: 'gmail',
                        auth: {
                            user: 'c9smartattendance@gmail.com',
                            pass: 'lostin2002'
                        }
                    });
        
                    let mailDetails = {
                        from: 'c9smartattendance@gmail.com',
                        to: `${results[0].email}`,
                        subject: `[${req.body.date} - ${req.body.sid}] Discrepancy Alert!`,
                        text: `${req.body.fid},\n\nYou have a new message from student (with roll no.) ${req.body.sid} regarding discrepancy in giving attendance.\nPlease check your message inbox in the Smart Attendence Portal for more details.\n\n\nRegards,\nSmart Attendance Manager\n\n\nNote: This is an auto-generated mail. Please do not send replies. Contact concerned staff or the department if any query.`
                    };
        
                    mailTransporter.sendMail(mailDetails, function(err, data) {
                        if(err) {
                            console.log('Error Occurs');
                        } else {
                            console.log('Email sent successfully');
                        }
                    });
                    // return res.json({
                    //     data: results
                    // })
                }
            })

            res.send("Success")
            console.log("Successfully added message")
        }
    })
})

const SELECT_ALL_FACULTIES_QUERY = 'SELECT * FROM faculty';
app.get('/', (req, res) => { res.send("Hello from server") });
app.get('/faculties', (req, res) => {
    connection.query(SELECT_ALL_FACULTIES_QUERY, (err, results) => {
        if (err) {
            return res.send(err)
        }
        else {
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculty/modifymail', (req, res) => {
    var {id,mail}=req.query;
    const FACULTY_MODIFYMAIL = `UPDATE f_email set email='${mail}' where id like '${id}'`;
    connection.query(FACULTY_MODIFYMAIL, (err, results) => {
        if (err) {
            return res.send(err)
        }
        else {
            console.log("Modified faculty mail")
            // return res.json({
            //     data: results
            // })
        }
    })
})

app.get('/student/modifymail', (req, res) => {
    var {id,mail}=req.query;
    const STUDENT_MODIFYMAIL = `UPDATE s_email set email='${mail}' where id like '${id}'`;
    connection.query(STUDENT_MODIFYMAIL, (err, results) => {
        if (err) {
            return res.send(err)
        }
        else {
            console.log("Modified student mail")
            // return res.json({
            //     data: results
            // })
        }
    })
})

const SELECT_ALL_COURSES_QUERY = 'SELECT * FROM course';
app.get('/', (req, res) => { res.send("Hello from server") });
app.get('/courses', (req, res) => {
    connection.query(SELECT_ALL_COURSES_QUERY, (err, results) => {
        if (err) {
            return res.send(err)
        }
        else {
            return res.json({
                data: results
            })
        }
    })
})

app.get('/students/add', (req, res) => {
    var { id, name, username, password, elective, department, section, semester } = req.query;
    elective = (elective === '' ? null : `'${elective}'`)
    const INSERT_STUDENT_QUERY = `INSERT INTO student values ('${id}','${name}','${username}','${password}',${elective},'${department}','${section}',${semester})`
    console.log(id, name, username, password, elective, department, section, semester)
    connection.query(INSERT_STUDENT_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("Success")
            console.log("Successfully added Student")
        }
    })
})

app.get('/students/add/mail', (req, res) => {
    var { id,mail} = req.query;
    const INSERT_FACULTY_QUERY2 = `INSERT INTO s_email values ('${id}','${mail}')`
    // console.log(id, name, username, password, department)
    connection.query(INSERT_FACULTY_QUERY2, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            // return res.json({
            //     data: results
            // })
            console.log("Successfully added student mail")
        }
    })
})


app.get('/faculties/add', (req, res) => {
    var { id, name, username, password, department } = req.query;
    const INSERT_FACULTY_QUERY = `INSERT INTO faculty values ('${id}','${name}','${username}','${password}','${department}')`
    console.log(id, name, username, password, department)
    connection.query(INSERT_FACULTY_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            // return res.json({
            //     data: results
            // })
            console.log("Successfully added faculty")
        }
    })
})

app.get('/faculties/add/mail', (req, res) => {
    var { id,mail} = req.query;
    const INSERT_FACULTY_QUERY2 = `INSERT INTO f_email values ('${id}','${mail}')`
    // console.log(id, name, username, password, department)
    connection.query(INSERT_FACULTY_QUERY2, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            // return res.json({
            //     data: results
            // })
            console.log("Successfully added faculty mail")
        }
    })
})

app.get('/admin/student', (req, res) => {
    const ADMIN_QUERY1 = `SELECT count(student.id) as ct from student`
    connection.query(ADMIN_QUERY1, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully retrieved the count")
        }
    })
})

app.get('/admin/faculty', (req, res) => {
    const ADMIN_QUERY2 = `SELECT count(faculty.id) as ct from faculty`
    connection.query(ADMIN_QUERY2, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully retrieved the count")
        }
    })
})

app.get('/admin/course', (req, res) => {
    const ADMIN_QUERY3 = `SELECT count(course.id) as ct from course`
    connection.query(ADMIN_QUERY3, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully retrieved the count")
        }
    })
})

app.get('/admin/msg', (req, res) => {
    const ADMIN_QUERY3 = `SELECT count(MsgId) as ct from msg where ack=0`
    connection.query(ADMIN_QUERY3, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully retrieved the count")
        }
    })
})

app.get('/student/home', (req, res) => {
    var { sid } = req.query;
    const STUDENT_HOME = `SELECT S_Name from student where id like '${sid}'`
    connection.query(STUDENT_HOME, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully retrieved the count")
        }
    })
})

app.get('/faculty/home', (req, res) => {
    var { fid } = req.query;
    const FACULTY_HOME = `SELECT F_Name from faculty where id like '${fid}'`
    connection.query(FACULTY_HOME, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log("Successfully displayed the content")
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculty/home/graph', (req, res) => {
    var { fid } = req.query;
    const FACULTY_HOME = `select CId,avg(percent) as av from (select CId,FId,class_attended*100/Total_classes as percent from attendance) as att where att.FId like '${fid}' group by att.CId`
    connection.query(FACULTY_HOME, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully retrieved the count")
        }
    })
})

app.get('/student/home/graph', (req, res) => {
    var { fid } = req.query;
    const FACULTY_HOME = `select CId,avg(percent) as av from (select CId,SId,class_attended*100/Total_classes as percent from attendance) as att where att.SId like '${fid}' group by att.CId`
    connection.query(FACULTY_HOME, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully retrieved the count")
        }
    })
})

app.get('/faculty/stats', (req, res) => {
    const FACULTY_STATS = `select Ack,count(*) as cnt from msg group by Ack order by Ack`
    connection.query(FACULTY_STATS, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculty/stats', (req, res) => {
    const FACULTY_STATS = `select Ack,count(*) as cnt from msg group by Ack order by Ack`
    connection.query(FACULTY_STATS, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/courses/add', (req, res) => {
    var { id, name, semester, type } = req.query;
    const INSERT_COURSE_QUERY = `INSERT INTO course values ('${id}','${name}','${semester}','${type}')`
    console.log(id, name, semester, type)
    connection.query(INSERT_COURSE_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
            console.log("Successfully added course")
        }
    })
})

app.get('/assign', (req, res) => {
    var { fid, cid, dept, sec, sem } = req.query;
    console.log(fid, cid, dept, sec, sem)
    const SELECT_COURSE = `SELECT C_type from course where id like '${cid}'`
    connection.query(SELECT_COURSE, (err, results) => {
        if (err) {
            console.log(err)
        }
        else {
            if (results[0].C_type === "Elective") {
                const SELECT_ATTENDANCE_QUERY = `SELECT id from student where C_elective like '${cid}' and S_Dept like '${dept}'`
                connection.query(SELECT_ATTENDANCE_QUERY, (err, results) => {
                    if (err) {
                        // return res.send(err)
                        console.log(err)
                    }
                    else {
                        console.log(results)
                        var len = results.length;
                        while (len > 0) {
                            len--;
                            const INSERT_ATTENDANCE_QUERY = `INSERT INTO attendance values ('${results[len].id}','${fid}','${cid}',0,0)`
                            connection.query(INSERT_ATTENDANCE_QUERY, (err, results) => {
                                if (err) {
                                    // return res.send(err)
                                    console.log(err)
                                }
                                else {
                                    console.log("Successfully added into attendance")
                                }
                            })
                        }
                    }
                })
            }
            else {
                const SELECT_ATTENDANCE_QUERY = `SELECT id from student where S_Dept='${dept}' and S_Sec='${sec}' and S_Sem='${sem}'`
                connection.query(SELECT_ATTENDANCE_QUERY, (err, results) => {
                    if (err) {
                        // return res.send(err)
                        console.log(err)
                    }
                    else {
                        console.log(results)
                        var len = results.length;
                        while (len > 0) {
                            len--;
                            const INSERT_ATTENDANCE_QUERY = `INSERT INTO attendance values ('${results[len].id}','${fid}','${cid}',0,0)`
                            connection.query(INSERT_ATTENDANCE_QUERY, (err, results) => {
                                if (err) {
                                    // return res.send(err)
                                    console.log(err)
                                }
                                else {
                                    console.log("Successfully added into attendance")
                                }
                            })
                        }
                    }
                })
            }
            console.log(results[0].C_type)
        }
    })

})

app.get('/students/modify', (req, res) => {
    var { id, name, username, password, elective, department, section, semester } = req.query;
    elective = (elective === 'null' ? null : `'${elective}'`)
    const MODIFY_STUDENT_QUERY = `UPDATE student SET S_Name='${name}',S_Username='${username}',S_Password='${password}',C_elective=${elective},S_Dept='${department}',S_Sec='${section}',S_Sem=${semester} where id like '${id}'`
    console.log(id, name, username, password, elective, department, section, semester)
    connection.query(MODIFY_STUDENT_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("success")
            console.log("Successfully modified student")
        }
    })
})

app.get('/faculties/modify', (req, res) => {
    var { id, name, username, password, department } = req.query;
    const MODIFY_FACULTY_QUERY = `UPDATE faculty SET F_Name='${name}',F_Username='${username}',F_Password='${password}',F_Dept='${department}' where id like '${id}'`
    console.log(id, name, username, password, department)
    connection.query(MODIFY_FACULTY_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("Success");
            console.log("Successfully modified student")
        }
    })
})

app.get('/updateattendance/od', (req, res) => {
    var { sid, fid, cid } = req.query;
    const UPDATEATTENDANCE_OD = `UPDATE attendance SET Class_attended=Class_attended+1 where FId='${fid}' and SId='${sid}' and CId='${cid}'`
    connection.query(UPDATEATTENDANCE_OD, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("Success");
            console.log("Successfully modified attendance table")
        }
    })
})

app.get('/updateattend/od', (req, res) => {
    var { sid, fid, cid, date } = req.query;
    const UPDATEATTEND_OD = `UPDATE attend SET Class_attended=1 where FId='${fid}' and SId='${sid}' and CId='${cid}' and Dt like '${date}'`
    connection.query(UPDATEATTEND_OD, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("Success");
            console.log("Successfully updated attend table")
            console.log(date)
        }
    })
})

app.get('/courses/modify', (req, res) => {
    var { id, name, type, sem } = req.query;
    const MODIFY_COURSES_QUERY = `UPDATE course SET C_Name='${name}',C_Type='${type}',C_Sem=${sem} where id like '${id}'`
    console.log(id, name, sem, type)
    connection.query(MODIFY_COURSES_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("Success");
            console.log("Successfully modified course")
        }
    })
})

app.get('/students/delete', (req, res) => {
    var { id } = req.query;
    const DELETE_STUDENT_QUERY = `DELETE FROM STUDENT where id like '${id}'`
    connection.query(DELETE_STUDENT_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("Success")
            console.log(
                "Successfully deleted student"
            )
        }
    })
})

app.get('/faculties/delete', (req, res) => {
    var { id } = req.query;
    const DELETE_FACULTY_QUERY = `DELETE FROM faculty where id like '${id}'`
    connection.query(DELETE_FACULTY_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            res.send("Success")
            console.log(
                "Successfully deleted faculty"
            )
        }
    })
})

app.get('/courses/delete', (req, res) => {
    var { id } = req.query;
    const DELETE_COURSE_QUERY = `DELETE FROM course where id like '${id}'`
    connection.query(DELETE_COURSE_QUERY, (err, results) => {
        if (err) {
            res.send(err)
            console.log(err)
        }
        else {
            res.send("Success")
            console.log(
                "Successfully deleted course"
            )
        }
    })
})

app.get('/students/login', (req, res) => {
    var { uname, pass } = req.query;
    const STUDENT_LOGIN_QUERY = `SELECT * FROM student where S_Username like '${uname}'`
    connection.query(STUDENT_LOGIN_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculties/login', (req, res) => {
    var { uname, pass } = req.query;
    const FACULTY_LOGIN_QUERY = `SELECT * FROM faculty where F_Username like '${uname}'`
    connection.query(FACULTY_LOGIN_QUERY, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculties/retrieve', (req, res) => {
    var { fid, cid, sem, sec } = req.query;
    const FACULTY_RETRIEVE = `select student.* from student,attendance where student.id=attendance.SId and attendance.FId like '${fid}' and attendance.CId like '${cid}' and student.S_Sem=${sem} and student.S_Sec like '${sec}'`
    connection.query(FACULTY_RETRIEVE, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/attend/retrieve', (req, res) => {
    var { sid, fid, cid } = req.query;
    const ATTEND_RETRIEVE = `select Dt,Class_attended from attend where SId like '${sid}' and CId like '${cid}'`
    connection.query(ATTEND_RETRIEVE, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/msgfaculty/retrieve', (req, res) => {
    var { sid } = req.query;
    const MSGFACULTY_RETRIEVE = `select distinct FId from attendance where SId like '${sid}'`
    connection.query(MSGFACULTY_RETRIEVE, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/msgstudent/retrieve', (req, res) => {
    var { fid } = req.query;
    const MSGSTUDENT_RETRIEVE = `select SId,MsgId,Msg,Doc,Dt,Ack from msg where FId like '${fid}'`
    connection.query(MSGSTUDENT_RETRIEVE, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/msg/ack', (req, res) => {
    var { fid } = req.query;
    const MSG_ACK = `update msg set ack=1 where MsgId like '${fid}'`
    connection.query(MSG_ACK, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
        }
    })
})

app.get('/attend/retrievecourse', (req, res) => {
    var { sid, sem } = req.query;
    if (sem !== null) {
        const ATTEND_RETRIEVECOURSE = `select CId from attendance where SId like '${sid}' and CId in (Select id from course where C_Sem=${sem})`
        connection.query(ATTEND_RETRIEVECOURSE, (err, results) => {
            if (err) {
                // return res.send(err)
                console.log(err)
            }
            else {
                console.log(results)
                return res.json({
                    data: results
                })
            }
        })
    }
})


app.get('/student/retrievemsg', (req, res) => {
    var { sid } = req.query;
    const STUDENT_RETRIEVEMSG = `select faculty.F_Name,msg.Msg,msg.Dt,msg.Doc,msg.Ack from faculty,msg where msg.FId like faculty.id and msg.SId like '${sid}'`
    connection.query(STUDENT_RETRIEVEMSG, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculties/adddate', (req, res) => {
    var { sid, fid, cid, date, clsattended } = req.query;
    const FACULTY_ADDDATE = `INSERT into attend values (${0},'${sid}','${fid}','${cid}','${date}','${clsattended}')`
    connection.query(FACULTY_ADDDATE, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/students/viewattendance', (req, res) => {
    var { sid, sem } = req.query;
    const STUDENT_VIEW_ATTENDANCE = `select course.id,course.C_Name,attendance.Class_attended,attendance.Total_classes,attendance.Class_attended*100/Total_classes as percent from course,attendance where attendance.SId='${sid}' and course.C_Sem=${sem} and attendance.CId= course.id`
    connection.query(STUDENT_VIEW_ATTENDANCE, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/admin/stats', (req, res) => {
    const ADMIN_STATS = `select CId,avg(Class_attended*100/Total_classes)as av from attendance group by CId`
    connection.query(ADMIN_STATS, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results)
            return res.json({
                data: results
            })
        }
    })
})



app.get('/faculties/updateclassesattended', (req, res) => {
    var { sid, fid, cid } = req.query;
    const FACULTY_UPDATE = `update attendance set Class_attended=Class_attended+1 where SId like '${sid}' and FId like '${fid}' and CId like '${cid}'`
    connection.query(FACULTY_UPDATE, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            // console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculties/updatetotalclasses', (req, res) => {
    var { sid, fid, cid } = req.query;
    const FACULTY_UPDATETOTAL = `update attendance set Total_classes=Total_classes+1 where SId like '${sid}' and FId like '${fid}' and CId like '${cid}'`
    connection.query(FACULTY_UPDATETOTAL, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            // console.log(results)
            return res.json({
                data: results
            })
        }
    })
})

app.get('/faculties/email', (req, res) => {
    var { id,date,cid } = req.query;
    const FACULTY_UPDATETOTAL = `select email from s_email where id like '${id}'`
    connection.query(FACULTY_UPDATETOTAL, (err, results) => {
        if (err) {
            // return res.send(err)
            console.log(err)
        }
        else {
            console.log(results[0].email)
            // var { sid, fid, cid } = req.query;
            let mailTransporter = nodemailer.createTransport({
                service: 'gmail',
                auth: {
                    user: 'c9smartattendance@gmail.com',
                    pass: 'lostin2002'
                }
            });

            let mailDetails = {
                from: 'c9smartattendance@gmail.com',
                to: `${results[0].email}`,
                subject: `Class Missed on ${date}`,
                text: `${id},\n\nYou have been marked absent for your class (Course Code: ${cid}) on ${date}.\n\nDon't forget to catch up on the missed class :).\n\n\nPlease contact respective faculty if any discrepancy.\n\n\nRegards,\nSmart Attendance Manager\n\n\nNote: This is an auto-generated mail. Please do not send replies. Contact concerned faculty or the department if any query.`
            };

            mailTransporter.sendMail(mailDetails, function(err, data) {
                if(err) {
                    console.log('Error Occurs');
                } else {
                    console.log('Email sent successfully');
                }
            });
            // return res.json({
            //     data: results
            // })
        }
    })

})



module.exports = app.listen(4000, () => { console.log("server listening on 4000") });