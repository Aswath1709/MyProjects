import '../index.css'
import Navbar from './navbar';
import Button from '@material-ui/core/Button';
import { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import ReactFontLoader from 'react-font-loader';
import * as React from 'react';
import Swal from 'sweetalert2'



const FacultyAdd = ({setFunc}) => {

    const [id, setId] = React.useState('')
    const [name, setName] = React.useState('')
    const [username, setUsername] = React.useState('')
    const [pass, setPass] = React.useState('')
    const [dept, setDept] = React.useState('')
    const [mail, setMail] = React.useState('')
   
    const submitHandler=(e)=>{
        e.preventDefault()
        console.log(id)
        fetch(`http://localhost:4000/faculties/add?id=${id}&name=${name}&username=${username}&password=${pass}&department=${dept}`)
        .then(response=>response.json())
        .catch(err=>console.error(err))

        fetch(`http://localhost:4000/faculties/add/mail?id=${id}&mail=${mail}`)
        // .then(response=>response.json())
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
            setId(event.target.value);
        }

        const handleChangeName=(event)=>{
            setName(event.target.value);
        }

        const handleChangeUsername=(event)=>{
            setUsername(event.target.value);
        }

        const handleChangePass=(event)=>{
            setPass(event.target.value);
        }

        const handleChangeDept=(event)=>{
            setDept(event.target.value);
        }

        const handleChangeMail=(event)=>{
            setMail(event.target.value);
        }

    return (
    <div>
     <Navbar />
     <form onSubmit={handler} className="Add-form">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'2%', backgroundColor:'rgb(60,60,60)', marginTop:'2%', fontFamily:'Kosugi Maru'}}>Go Back</Button>
            </form>
     <form className="Add-form">
         <h4 className="form-header">Add Faculty</h4>
         <div style={{display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'center',width:'100%'}}>
         <div className="form-group">
                <label className="login-form-label">Facuty Id</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={id} onChange={handleChangeRno} placeholder="F01"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Faculty Name</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={name} onChange={handleChangeName} placeholder="Faculty Name"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Faculty Email</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={mail} onChange={handleChangeMail} placeholder="abc@gmail.com"></input>
            </div>          
            <div className="form-group">
                <label className="login-form-label">Faculty Username</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={username} onChange={handleChangeUsername} placeholder="Faculty Username"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Password</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={pass} onChange={handleChangePass} placeholder="Password"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Department</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={dept} onChange={handleChangeDept} placeholder="CSE"></input>
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
      );
}
 


export default FacultyAdd;