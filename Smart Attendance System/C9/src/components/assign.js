import React from 'react'
import Navbar from './navbar'
import '../index.css'
import Button from '@material-ui/core/Button';
import { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import ReactFontLoader from 'react-font-loader';
import Swal from 'sweetalert2';


const Assign = () => {

    const [fid, setFid] = React.useState('')
    const [cid, setCid] = React.useState('')
    const [dept, setDept] = React.useState('')
    const [sec, setSec] = React.useState('')
    const [sem, setSem] = React.useState('')
   
    const submitHandler=(e)=>{
        e.preventDefault()
        fetch(`http://localhost:4000/assign?fid=${fid}&cid=${cid}&dept=${dept}&sec=${sec}&sem=${sem}`)
        .then(response=>response.json())
        .catch(err=>console.error(err))

        Swal.fire({
            title: 'Success!',
            text: 'Do you want to continue',
            icon: 'success',
            confirmButtonText: 'Done'
          })
        }
        
        const handler=(e)=>{
            e.preventDefault();
            window.location='/admin/add'
        }    

        const handleChangeRno=(event)=>{
            setFid(event.target.value);
        }

        const handleChangeName=(event)=>{
            setCid(event.target.value);
        }

        const handleChangeUsername=(event)=>{
            setDept(event.target.value);
        }

        const handleChangePass=(event)=>{
            setSec(event.target.value);
        }

        const handleChangeDept=(event)=>{
            setSem(event.target.value);
        }


    return (
        <div>
            <Navbar/>
            <form onSubmit={handler} className="Add-form">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'2%', backgroundColor:'rgb(60,60,60)', marginTop:'2%', fontFamily:'Kosugi Maru'}}>Go Back</Button>
            </form>
        <form className="Add-form">
         <h4 className="form-header">Assign Faculty</h4>
         <div style={{display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'center',width:'100%'}}>
         <div className="form-group">
                <label className="login-form-label">Faculty Id</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={fid} onChange={handleChangeRno} placeholder="Faculty Id"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Course Id</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={cid} onChange={handleChangeName} placeholder="Course Id"></input>
            </div>          
            <div className="form-group">
                <label className="login-form-label">Department</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={dept} onChange={handleChangeUsername} placeholder="Department"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Section</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={sec} onChange={handleChangePass} placeholder="Section"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Semester</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={sem} onChange={handleChangeDept} placeholder="Semester"></input>
            </div>
            </div>
            {/* <p>{id}</p>
            <p>{name}</p>
            <p>{username}</p>
            <p>{pass}</p>
            <p>{dept}</p> */}
            <Button onClick={submitHandler} type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'47%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Submit</Button>
     </form>
        </div>
    )
}

export default Assign
