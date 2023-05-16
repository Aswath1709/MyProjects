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
import Swal from 'sweetalert2';


const AdminAdd = ({setFunc}) => {

    const [id, setId] = React.useState('')
    const [name, setName] = React.useState('')
    const [sem, setSem] = React.useState(1)
    const [type, setType] = React.useState('Elective')
   
    const submitHandler=(e)=>{
        e.preventDefault()
        fetch(`http://localhost:4000/courses/add?id=${id}&name=${name}&semester=${sem}&type=${type}`)
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
        setId(event.target.value);
    }

    const handleChangeName=(event)=>{
        setName(event.target.value);
    }

    const handleChangeSem=(event)=>{
        setSem(event.target.value);
    }

    const handleChangeType=(event)=>{
        setType(event.target.value);
    }

    return (
    <div>
     <Navbar />
     <form onSubmit={handler} className="Add-form">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'2%', backgroundColor:'rgb(60,60,60)', marginTop:'2%', fontFamily:'Kosugi Maru'}}>Go Back</Button>
            </form>
     <form className="Add-form">
         <h4 className="form-header">Add Course</h4>
         <div style={{display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'center',width:'100%'}}>
         <div className="form-group">
                <label className="login-form-label">Course Id</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={id} onChange={handleChangeRno} placeholder="CourseId"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Course Name</label>
                <input type="text" style={{textAlign:'center'}} className="form-control login-form-input" value={name} onChange={handleChangeName} placeholder="Course Name"></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Semester</label>
                <select id = "myList" className="form-control login-form-input" value={sem} onChange={handleChangeSem}>
               <option value = "1">1</option>
               <option value = "2">2</option>
               <option value = "3">3</option>
               <option value = "4">4</option>               
               <option value = "5">5</option>
               <option value = "6">6</option>
               <option value = "7">7</option>
               <option value = "8">8</option>
             </select>
            </div>
            <div className="form-group">
                <label className="login-form-label">Course Type</label>
                <select id = "myList" className="form-control login-form-input" value={type} onChange={handleChangeType}>
               <option value = "Elective">Elective</option>
               <option value = "Core">Core</option>
             </select>
             
            </div>
            </div>
            <Button type="submit" onClick={submitHandler} class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'47%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Submit</Button>
            
            </form>
            
    </div>
      );
}
 


export default AdminAdd;