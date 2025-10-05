import '../index.css'
import Navbar from './user-navbar';
import Button from '@material-ui/core/Button';
import { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import ReactFontLoader from 'react-font-loader';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
import React from 'react';
import TextField from '@material-ui/core/TextField';
import Select from '@material-ui/core/Select';
import MenuItem from '@material-ui/core/MenuItem';
import { useParams } from 'react-router-dom';
import NotAuth from './notAvailable';
import background1 from '../assets/AttendBackground1.png'
import background2 from '../assets/AttendBackground2.png'
import background3 from '../assets/AttendBackground3.png'
var CryptoJS = require("crypto-js");



const useStyles = makeStyles((theme) => ({
  container: {
    display: 'flex',
    flexWrap: 'wrap',
  },
  textField: {
    marginTop: theme.spacing(5),
    // marginLeft: theme.spacing(5),
    marginRight: theme.spacing(1),
    width: '100%',
  },
}));



const AddAttendance = ({ setFunc }) => {
  const [courses, setCourses] = React.useState(null);
  async function getcourses() {
    fetch("http://localhost:4000/courses")
      .then(response => response.json())
      .then(response => setCourses(response.data))
      .catch(err => console.error(err))
  }
  React.useEffect(() => {
    getcourses()
    async function authfn() {
      if (local) {
        const response = await fetch(`http://localhost:4000/faculties/auth?id=${id}`)
        const json = await response.json()
        // console.log(json.data[0])
        var bytes = CryptoJS.AES.decrypt(local, 'my-secret-key@123');
        var decr = JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
        // console.log(decr[0].pass===json.data[0].S_Password)
        var v = decr[0].uname === json.data[0].F_Username && decr[0].pass === json.data[0].F_Password ? 1 : 0
        setAuth(v)
      }



    }
    authfn()
  }, [])
  var { id } = useParams()

  const [cid, setCid] = React.useState("");
  const [sem, setSem] = React.useState("");
  const [sec, setSec] = React.useState("");
  const [students, setStudents] = React.useState("");
  const [showtable, setShowtable] = React.useState(0);
  const [date, setDate] = React.useState((new Date()).getFullYear() + '-' + '0' + ((new Date()).getMonth() + 1) + '-' + (new Date()).getDate());
  const [local, setLocal] = React.useState(localStorage.getItem('user') || null)
  const [auth, setAuth] = React.useState(0)
  // var x=students.map((student,index)=>{

  //     student.id:false,

  // })
  const classes = useStyles();
  var obj = {}

  const [state, setState] = React.useState(obj);
  console.log(state)
  const handleChange = (event) => {
    setState({ ...state, [event.target.name]: event.target.checked });
    console.log(state)
  };

  const onDateChange = (e) => {
    setDate(e.target.value)
  }

  const submitHandlerMain = (e) => {
    e.preventDefault();
    var ob
    var obj2
    for (var i = 0; i < students.length; i++) {
      obj2 = { [students[i].id]: false }
      ob = { ...ob, ...obj2 }
      console.log(ob)
    }

    fetch(`http://localhost:4000/faculties/retrieve?fid=${id}&cid=${cid}&sem=${sem}&sec=${sec}`)
      .then(response => response.json())
      .then(response => { setStudents(response.data) })
      .catch(err => console.error(err))
    console.log(students)
    setShowtable(1)
  }

  const handler = (e) => {
    e.preventDefault();
    window.location = `/faculty/${id}`
  }
  const updateHandler = (e) => {
    e.preventDefault();
    for (var key in state) {
      if (state.hasOwnProperty(key)) {
        if (state[key] == true) {
          fetch(`http://localhost:4000/faculties/updateclassesattended?sid=${key}&fid=${id}&cid=${cid}`)
            .then(response => response.json())
            .catch(err => console.error(err))
        }
        if (state[key] === false) {
          fetch(`http://localhost:4000/faculties/email?id=${key}&date=${date}&cid=${cid}`)
            // .then(response => response.json())
            .catch(err => console.error(err))
        }


        fetch(`http://localhost:4000/faculties/updatetotalclasses?sid=${key}&fid=${id}&cid=${cid}`)
          .then(response => response.json())
          .catch(err => console.error(err))
        // alert("Successfully Updated!");
        // window.location = '/add-attendance'
      }
    }

    for (var key in state) {
      if (state.hasOwnProperty(key)) {
        if (state[key] == true) {
          fetch(`http://localhost:4000/faculties/adddate?sid=${key}&fid=${id}&cid=${cid}&date=${date}&clsattended=${1}`)
            .then(response => response.json())
            .catch(err => console.error(err))
        }
        else {
          fetch(`http://localhost:4000/faculties/adddate?sid=${key}&fid=${id}&cid=${cid}&date=${date}&clsattended=${0}`)
            .then(response => response.json())
            .catch(err => console.error(err))
        }
      }
    }
  }

  const handleChangeSec = (e) => {
    setSec(e.target.value)
  }

  const handleChangeSem = (e) => {
    setSem(e.target.value)
  }

  const handleChangeCid = (e) => {
    setCid(e.target.value)
  }


  return (
    <div>{auth === 0 ? (<div>
      <NotAuth />
    </div>) :
      (<div>
        <Navbar message={"faculty"} fid={id} />
        <div style={{ background: `url(${background1})`,
        height: '91.5vh',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        backgroundSize:'cover'}}>
        <form onSubmit={handler} className="Add-form">
          <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{ height: '100%', marginLeft: '2%', backgroundColor: 'rgb(60,60,60)', marginTop: '2%', fontFamily: 'Kosugi Maru' }}>Go Back</Button>
        </form>
        <form style={{ display: "flex", alignItems: 'center', justifyContent: 'center', flexDirection: "column" }} onSubmit={submitHandlerMain} className="Add-form">
          <h4 className="form-header">Comprehensive Attendance Report</h4>
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
            <label className="login-form-label">Semester</label>
            <select value={sem} style={{ marginLeft: '0', width: '12vw', textAlign: 'center' }} onChange={handleChangeSem} id="Course" className="form-attendance">
              <option value="0">Select</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5</option>
              <option value="6">6</option>
              <option value="7">7</option>
              <option value="8">8</option>
            </select>
            <label className="login-form-label">Course</label>
            {courses && <select style={{ marginLeft: '0', width: '12vw' }} value={cid} onChange={handleChangeCid} id="Course" className="form-attendance">
              <option value="0">Select</option>
              {courses.map((name, index) => (
                <option key={index} value={name.id}>
                  {name.id}
                </option>
              ))}

            </select>}


            <label className="login-form-label">Section</label>
            <input style={{ marginLeft: '0', width: '12vw' }} type="text" value={sec} onChange={handleChangeSec} id="" className="form-attendance" placeholder="Section"></input>

            <TextField
              id="date"
              label="Date"
              type="date"
              value={date}
              onChange={onDateChange}
              // defaultValue='2021-04-29'
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
            />
            {console.log(date)}

          </div>

          <div id="update-btn">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{ height: '100%', marginTop: '2vh', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>View</Button>
          </div>
        </form>
        <form onSubmit={updateHandler}>
          {showtable === 1 && (
            <div id="student-info">
              {students && <table>

                <tr>
                  <th>Roll Number</th>
                  <th>Name</th>
                  <th>Attendance</th>
                </tr>
                {students.map((student, index) => (
                  <tr key={index}>
                    <td>{student.id}</td>
                    <td>{student.S_Name}</td>
                    <td >
                      <Switch
                        checked={state.checkedA}
                        onChange={handleChange}
                        color="primary"
                        name={student.id}
                        inputProps={{ 'aria-label': 'primary checkbox' }}
                      />
                    </td>
                  </tr>
                ))}
              </table>
              }
              <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{ height: '100%', marginLeft: '47%', marginTop: '2%', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>Update</Button>
            </div>)}
        </form>
      </div>
    </div>)
}
    </div >
  );
}



export default AddAttendance;