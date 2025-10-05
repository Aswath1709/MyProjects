import Navbar from './navbar';
import Button from '@material-ui/core/Button';
import * as React from 'react';
import { DataGrid, GRID_ROW_SELECTED } from '@material-ui/data-grid';
import DeleteIcon from '@material-ui/icons/Delete';
import { makeStyles } from '@material-ui/core/styles';
import TextField from '@material-ui/core/TextField';
import {useEffect} from 'react';


const columns = [
    { field: 'id', headerName: 'ID', width: 200 },
    { field: 'S_Name', headerName: 'Student Name', width: 130 },
    { field: 'S_Username', headerName: 'Username', width: 130 },
    { field: 'S_Password', headerName: 'Password', width: 130 },
    { field: 'C_elective', headerName: 'Elective', width: 130 },
    { field: 'S_Dept', headerName: 'S_Dept', width: 130 },
    { field: 'S_Sec', headerName: 'Section', width: 130 },
    {
      field: 'S_Sem',
      headerName: 'Sem',
      type: 'number',
      width: 90,
    },
  ];

const StudentDelete = ({ Students }) => {
    const [students, setStudents] = React.useState(Students);
        async function getstudents(){
          fetch("http://localhost:4000/students")
          .then(response=>response.json())
          .then(response=>setStudents(response.data))
          .catch(err=>console.error(err))
        }
        useEffect(()=>{
        getstudents()
    },[students])
    const [rowind,setRowind]=React.useState(null)
    const rows = students
    const handleclick=(e)=>{
        setRowind(e.rowIndex)
    }

   

    
    
    const handleDelete=()=>{
        console.log(rowind) 
        console.log(rows[rowind].id)       
        fetch(`http://localhost:4000/students/delete?id=${rows[rowind].id}`)
        .then(response=>response.json())
        .catch(err=>console.error(err))    
    }

    return ( 
        <div>
            <Navbar/>
            <h3 style={{marginTop:'3vh',paddingLeft:'2vw',marginBottom:'3vh'}}><u>Delete Student</u></h3>
            <div style={{display:'flex',justifyContent:'center',alignItems:'center',flexDirection:'column'}}>
            <div style={{ height: '50vh', width: '80%',display:'flex',justifyContent:'center',alignItems:'center',marginBottom:'3vh' }}>
            <DataGrid onRowClick={(event)=>handleclick(event)} rows={rows} columns={columns} pageSize={5} />
            </div>
            <Button onClick={handleDelete} variant="contained"><DeleteIcon/>Delete</Button>
            </div>
        </div>

     );
}
 
export default StudentDelete;