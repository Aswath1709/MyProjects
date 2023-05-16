import {BrowserRouter as Router, Route, Switch} from 'react-router-dom';
import FacultyHome from './facultyhome';
import Homemain from './homemain';
import StudentHome from './studenhome'
import AdminHome from './adminhome';
import AdminAdd from './adminAdd';
import FacultyAdd from './facultyAdd';
import CourseAdd from './courseAdd';
import StudentAdd from './studentAdd'
import Password from './forgot_pass';
import Email from './get_email';
import Logout from './logout';
import Modifystudent from './modifystudent';
import AdminModify from './adminModify';
import Modifyfaculty from './modifyfaculty';
import Modifycourse from './modifycourse';
import StudentDelete from './studentDelete';
import AdminDelete from './adminDelete';
import FacultyDelete from './facultyDelete';
import CourseDelete from './courseDelete';
import StudentAttendance from './studentAttendance';
import AddAttendance from './addAttendance';
import Assign from './assign';
import Studentmsg from './studentmessage';
import Facultymsg from './facultyMessage';
import UpdateAttendance from './updateAttendance';
import * as React from 'react'
import NotAuth from './notAvailable';
var CryptoJS = require("crypto-js");

const Students=[
  {
    'id':'Cse18237',
    'S_Name':'Manoj',
    'S_Username':'manojkumar',
    'S_Password':'manojkumar',
    'C_elective':'Open lab python',
    'S_Dept':'Computer Science',
    'S_Sec':'C',
    'S_Sem':'6',
  },

  {
    'id':'Cse18203',
    'S_Name':'Abishek',
    'S_Username':'abishek',
    'S_Password':'Vasanthan',
    'C_elective':'Open lab python',
    'S_Dept':'Computer Science',
    'S_Sec':'C',
    'S_Sem':'6',
  },

  {
    'id':'Cse18233',
    'S_Name':'Manoj',
    'S_Username':'manojkumar',
    'S_Password':'manojkumar',
    'C_elective':'Open lab java',
    'S_Dept':'Computer Science',
    'S_Sec':'C',
    'S_Sem':'4',
  }
  ,
  {
    'id':'Cse18246',
    'S_Name':'Roshni',
    'S_Username':'Rajesh',
    'S_Password':'I love manoj',
    'C_elective':'manoj',
    'S_Dept':'Computer Science',
    'S_Sec':'C',
    'S_Sem':'5',
  }
]

const Faculties=[
{
    'id':'asdfgf',
    'F_Name':'Manoj',
    'F_Username':'manojkumar',
    'F_Password':'manojkumar',
    'F_Dept':'Computer Science',
},

{
  'id':'lkjhj',
  'F_Name':'Abishek',
  'F_Username':'abishek',
  'F_Password':'vasanthan',
  'F_Dept':'ECE',
},

{
  'id':'qwerty',
  'F_Name':'Aswanth',
  'F_Username':'Aswanth',
  'F_Password':'Ragavendra',
  'F_Dept':'Mechanical',
}
]

const Courses=[
  {
      'id':'15CSE202',
      'C_Name':'DataStructures',
      'C_Sem':'3',
      'C_Type':'Core',
  },
  
  {
    'id':'15CSE432',
    'C_Name':'Machine Learning',
    'C_Sem':'6',
    'C_Type':'Core',
  },

  
  {
    'id':'15CSE336',
    'C_Name':'Biometrics',
    'C_Sem':'6',
    'C_Type':'Elective',
  }
]



const HomeRoute = () => {
  const [local,setLocal]=React.useState(localStorage.getItem('user')||null)
  const [auth,setAuth]=React.useState(0)
React.useEffect(()=>{
  function authfn() {
    if(local){
    var bytes = CryptoJS.AES.decrypt(local, 'my-secret-key@123');
    var decr = JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
    var v=decr[0].uname==='admin'&&decr[0].pass==='admin'?1:0
    setAuth(v)}



  }
authfn()
},[])  
    return (
        <Router>
        <div className="content">
          <Switch>
            <Route exact path="/">
              <Homemain />
            </Route>
            <Route exact path="/student/:id">
              <StudentHome />
            </Route>
            <Route exact path="/faculty/:id">
              <FacultyHome />
            </Route>
            <Route exact path="/admin">
              {auth===1?<AdminHome />:<NotAuth/>}
            </Route>
            <Route exact path="/attendance/:id">
              <StudentAttendance />
            </Route>
            <Route exact path="/add-attendance/:id">
              <AddAttendance />
            </Route>
            <Route exact path="/admin/add">
            {auth===1?<AdminAdd />:<NotAuth/>}
            </Route>
            <Route exact path="/admin/add/course">
            {auth===1?<CourseAdd />:<NotAuth/>}
            </Route>
            <Route exact path="/admin/add/faculty">
            {auth===1?<FacultyAdd />:<NotAuth/>}
            </Route>
            <Route exact path="/admin/add/student">
            {auth===1?<StudentAdd />:<NotAuth/>}
            </Route>
            <Route exact path="/password">
              <Password />
            </Route>
            <Route exact path="/email">
              <Email />
            </Route>
            <Route exact path="/logout">
              <Logout />
            </Route>
            <Route exact path="/admin/modify/student">
            {auth===1?<Modifystudent Students={Students}/>:<NotAuth/>}
            </Route>
             <Route exact path="/admin/modify/faculty">
              {auth===1?<Modifyfaculty Faculties={Faculties}/>:<NotAuth/>}
            </Route>
            <Route exact path="/admin/modify/course">
              {auth===1?<Modifycourse Courses={Courses}/>:<NotAuth/>}
            </Route> 
            <Route exact path="/admin/modify">
              {auth===1?<AdminModify/>:<NotAuth/>}
            </Route>
            <Route exact path="/admin/delete">
              {auth===1?<AdminDelete/>:<NotAuth/>}
            </Route>
            <Route exact path="/admin/delete/student">
              {auth===1?<StudentDelete Students={Students}/>:<NotAuth/>}
            </Route>
            <Route exact path="/admin/delete/faculty">
              {auth===1?<FacultyDelete Faculties={Faculties}/>:<NotAuth/>}
            </Route>
            <Route exact path="/admin/delete/course">
              {auth===1?<CourseDelete Courses={Courses}/>:<NotAuth/>}
            </Route>
            <Route exact path="/admin/assign">
              {auth===1?<Assign Courses={Courses}/>:<NotAuth/>}
            </Route>
            <Route exact path="/student/:id/msg">
              <Studentmsg/>
            </Route>
            <Route exact path="/faculty/:id/msg">
              <Facultymsg/>
            </Route>
            <Route exact path="/facultyupdateattendance/:id">
              <UpdateAttendance/>
            </Route>
          </Switch>
        </div>
    </Router>

      );
}
 
export default HomeRoute;