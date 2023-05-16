import '../index.css'
import Navbar from './user-navbar';
import Button from '@material-ui/core/Button';
import { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import ReactFontLoader from 'react-font-loader';
import * as React from 'react';
import { useParams } from 'react-router-dom';
import reactDom from 'react-dom';
import CheckIcon from '@material-ui/icons/Check';
import NotAuth from './notAvailable';
var CryptoJS = require("crypto-js");


const HandleClassMiss=({x,y})=>{
    const [clas, setClas] = React.useState(null)
    React.useEffect(()=>{setClas(Math.floor((4*x-3*y)/3))},[])
    return <p style={{fontSize:'0.75em',color:'green'}}>{`You cannot miss more than ${clas} classes`}</p>
}



const HandleClassAttend=({x,y})=>{
    const [clas, setClas] = React.useState(null)
    React.useEffect(()=>{setClas(Math.ceil(3*y-4*x))},[])
    return <p style={{fontSize:'0.75em',color:'red'}}>{`You must attend ${clas} more classes`}</p>
}

const StudentAttendance = ({ setFunc }) => {
    var { id } = useParams()
    const [sem, setSem] = React.useState('')
    const [sem1, setSem1] = React.useState('')
    const [course, setCourse] = React.useState('')
    const [crsarray, setCrsarray] = React.useState(null)
    const [show, setShow] = React.useState(0)
    const [val, setVal] = React.useState(null)
    const [val1, setVal1] = React.useState(null)
    const [show1, setShow1] = React.useState(false)
    const [formshow, setFormShow] = React.useState(false)
    const [formshow1, setFormShow1] = React.useState(false)
    const [local,setLocal]=React.useState(localStorage.getItem('user')||null)
    const [auth,setAuth]=React.useState(0)

    React.useEffect(()=>{
        async function authfn() {
            if(local){
            const response = await fetch(`http://localhost:4000/students/auth?id=${id}`)
            const json = await response.json()
            console.log(json.data[0])
            var bytes = CryptoJS.AES.decrypt(local, 'my-secret-key@123');
            var decr = JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
            console.log(decr[0].pass===json.data[0].S_Password)
            var v=decr[0].uname===json.data[0].S_Username&&decr[0].pass===json.data[0].S_Password?1:0
            setAuth(v)}
    
    
      
          }
    authfn()

    },[])

    const submitHandler = (e) => {
        e.preventDefault();
        fetch(`http://localhost:4000/students/viewattendance?sid=${id}&sem=${sem}`)
            .then(response => response.json())
            .then(response => setVal(response.data))
            .catch(err => console.error(err))
        setShow(1)
        console.log(val)
    }

    const submitHandler1 = (e) => {
        e.preventDefault();
        fetch(`http://localhost:4000/attend/retrieve?sid=${id}&cid=${course}`)
            .then(response => response.json())
            .then(response => setVal1(response.data))
            .catch(err => console.error(err))
        setShow1(true)

        console.log(val)
    }

    const handler = (e) => {
        e.preventDefault();
        window.location = `/student/${id}`
    }
    const handleChangeSem = (e) => {
        setSem(e.target.value)
    }

    const handleChangeSem1 = (e) => {
        setSem1(e.target.value)
        // fetch(`http://localhost:4000/attend/retrievecourse?sid=${id}&sem=${sem1}`)
        // .then(response=>response.json())
        // .then(response=>setCrsarray(response.data))
        // .catch(err=>console.error(err))
        // console.log(crsarray)
    }

    const fn=(e)=>{
        fetch(`http://localhost:4000/attend/retrievecourse?sid=${id}&sem=${sem1}`)
        .then(response=>response.json())
        .then(response=>setCrsarray(response.data))
        .catch(err=>console.error(err))
    }

    const handleChangeCourse = (e) => {
        setCourse(e.target.value)
    }

    const handleFormShow = () => {
        setFormShow(true)
        setFormShow1(false)
        setShow1(false)
    }

    const handleFormShow1 = () => {
        setShow(0)
        setFormShow(false)
        setFormShow1(true)
    }

    return (
        <div>{
            auth===0?(<div>
              <NotAuth/>
            </div>):
        (<div>
            <Navbar message="student" fid={id}/>
            <form onSubmit={handler} className="Add-form">
                <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{ height: '100%', marginLeft: '2%', backgroundColor: 'rgb(60,60,60)', marginTop: '2%', fontFamily: 'Kosugi Maru' }}>Go Back</Button>
            </form>
            <div className="Add-form">
                <h4 className="form-header">Comprehensive Attendance Report</h4>
                <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
                    <button onClick={handleFormShow} style={{ height: '100%', border: '0px', borderRadius: '4px', color: 'white', marginRight: '0.5vw', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>Attendance Summary</button>
                    <button onClick={handleFormShow1} style={{ height: '100%', border: '0px', borderRadius: '4px', color: 'white', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>Course Wise Report</button>
                </div>
                {formshow === true && (<div>
                    <div>
                        <label className="login-form-label">Semester</label>
                        <select value={sem} onChange={handleChangeSem} id="Course" className="form-attendance">
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
                    </div>
                    <div id="submit-btn">
                        <Button onClick={submitHandler} class="btn btn-primary" variant="contained" color="primary" disableElevation style={{ height: '100%', marginLeft: '47%', marginTop: '2%', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>Submit</Button>
                    </div>
                </div>)}
                {show === 1 && <div id="student-info">
                    {val && <table>

                        <tr>
                            <th>Course Id</th>
                            <th>Name</th>
                            <th>Classes Attended</th>
                            <th>Total Classes</th>
                            <th>Percentage</th>
                        </tr>
                        {
                            val.map((data, index) => (
                                <tr>
                                    <td>{data.id}</td>
                                    <td><div style={{display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'center'}}>
                                        <p>{data.C_Name}</p>
                                        {data.percent>=75?<HandleClassMiss x={data.Class_attended} y={data.Total_classes}/>:<HandleClassAttend x={data.Class_attended} y={data.Total_classes}/>}
                                        </div></td>
                                    <td>{data.Class_attended}</td>
                                    <td>{data.Total_classes}</td>
                                    <td>{data.percent}</td>
                                </tr>)
                            )
                        }
                    </table>}
                </div>}
                {formshow1 && <div id="student-info">
                <label className="login-form-label">Semester</label>
                    <div style={{display:'flex',flexDirection:'row'}}>
                    <select value={sem1} onChange={handleChangeSem1} id="Course" className="form-attendance">
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
                    <CheckIcon onClick={fn}/>
                    </div>

                    
                        
                    {console.log(course)}
                    {
                    crsarray!==null&&crsarray.length!==0&&(
                        <div>
                        <label className="login-form-label">Course</label>
                        <select value={course} onChange={handleChangeCourse} id="Course" className="form-attendance">
                        <option value="0">Select Course</option>
                        {crsarray.map((crs,index)=>
                        (<option key={index} value={`${crs.CId}`}>{crs.CId}</option>))}
                        
                        {/* <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6</option>
                        <option value="7">7</option>
                        <option value="8">8</option> */}
                    </select>
                    <Button onClick={submitHandler1} class="btn btn-primary" variant="contained" color="primary" disableElevation style={{ height: '100%', marginLeft: '47%', marginTop: '2%', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>View</Button>
                    </div>)}


                    
                    
                    {/* <input value={sem} onChange= type='text' className="form-attendance"></input> */}
                    {val1&&show1===true&&
            <table>
            
            <tr>
                <th>Date</th>
                <th>Classes Attended</th>
            </tr>
            {
                val1.map((data,index)=>(
                <tr>
                    <td>{data.Dt}</td>
                    <td>{data.Class_attended}</td>
                </tr>)
                )
            }
        </table>}
        
                </div>}
            </div>
        </div>)}
        </div>
    );
}



export default StudentAttendance;