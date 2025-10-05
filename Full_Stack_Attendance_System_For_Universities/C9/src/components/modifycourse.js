import Navbar from './navbar';
import Button from '@material-ui/core/Button';
import * as React from 'react';
import { DataGrid, GRID_ROW_SELECTED } from '@material-ui/data-grid';
import { makeStyles } from '@material-ui/core/styles';
import { Row } from 'react-bootstrap';
import TextField from '@material-ui/core/TextField';
import {useEffect} from 'react';

const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,
        // maxWidth: 752,
    },
    demo: {
        backgroundColor: theme.palette.background.paper,
    },
    title: {
        margin: theme.spacing(4, 0, 2),
    },
}));

const columns = [
    { field: 'id', headerName: 'ID', width: 200 },
    { field: 'C_Name', headerName: 'Course Name', width: 200 },
    { field: 'C_type', headerName: 'Type', width: 200 }, 
    {
      field: 'C_Sem',
      headerName: 'Semester',
      type: 'number',
      width: 200,
    },
     
  ];

  
//      {Courses.map((Student,index)=>{
//          {id:Student.S_Id, Name:Student.C_Name}
//      })}
// //     { id: 1, lastName: 'Snow', firstName: 'Jon', age: 35 },
// //     { id: 2, lastName: 'Lannister', firstName: 'Cersei', age: 42 },
// //     { id: 3, lastName: 'Lannister', firstName: 'Jaime', age: 45 },
// //     { id: 4, lastName: 'Stark', firstName: 'Arya', age: 16 },
// //     { id: 5, lastName: 'Targaryen', firstName: 'Daenerys', age: null },
// //     { id: 6, lastName: 'Melisandre', firstName: null, age: 150 },
// //     { id: 7, lastName: 'Clifford', firstName: 'Ferrara', age: 44 },
// //     { id: 8, lastName: 'Frances', firstName: 'Rossini', age: 36 },
// //     { id: 9, lastName: 'Roxie', firstName: 'Harvey', age: 65 },
//   ];

const Modifycourse = ({ Courses }) => {
    const classes = useStyles();
    async function getstudents(){
      fetch("http://localhost:4000/courses")
      .then(response=>response.json())
      .then(response=>setCourses(response.data))
      .catch(err=>console.error(err))
    }
    useEffect(()=>{
    getstudents()
})
    

    const handleclick=(e)=>{
      setRowind(e.rowIndex)
      setName(e.row.C_Name)
      setType(e.row.C_type)
      setSem(e.row.C_Sem)
      setId(e.row.id)
    }

    const [id, setId] = React.useState('');
    const [courses, setCourses] = React.useState(Courses);
    const [name, setName] = React.useState('');
    const [type, setType] = React.useState('');
    const [sem, setSem] = React.useState('');
    const [rowind,setRowind]=React.useState(null)

    const rows = courses
  
    const handleChangeName = (event) => {
    setName(event.target.value);
  };

  const handleChangeType = (event) => {
    setType(event.target.value);
  };

  const handleChangeSem = (event) => {
    setSem(event.target.value);
  };

  const handleChangeId = (event) => {
    setId(event.target.value);
  };

  const submitHandler=(event)=>{
    fetch(`http://localhost:4000/courses/modify?id=${id}&name=${name}&type=${type}&sem=${sem}`)
        .then(response=>response.json())
        .catch(err=>console.error(err))
    // console.log(rows)

  //   // setName(e.row.C_Name)
  //   // setRno(e.row.id)
  //   // setUname(e.row.C_Type)
  //   // setPass(e.row.S_Password)
  //   // setElective(e.row.C_elective)
  //   // setDept(e.row.C_Type)
  //   // setSem(e.row.C_Sem)
  }
    
    
    return (
      <div>
        <Navbar/>
        <h3 style={{marginTop:'3vh',paddingLeft:'2vw'}}><u>Modify Course</u></h3>
        <div style={{display:'flex',justifyContent:'center',alignItems:'center',flexDirection:'column'}}>
          <div style={{display:'flex',justifyContent:'center',alignItems:'center',marginTop:'3vh',marginBottom:'3vh'}}>
          <form onSubmit={submitHandler}>
          <TextField label="Roll No" disabled value={id} onChange={handleChangeId} style={{marginLeft:'2vh'}} />
          <TextField label="Name" value={name} onChange={handleChangeName} style={{marginLeft:'2vh'}}/>
          <TextField label="Semester" value={sem} onChange={handleChangeSem} style={{marginLeft:'2vh'}}/>
          <TextField label="Course Type" value={type} onChange={handleChangeType} style={{marginLeft:'2vh'}}/>
          <div  style={{display:'flex',justifyContent:'center',alignItems:'center'}}>
          <Button type='submit' variant="contained" style={{marginLeft:'2vh',marginTop:'2vh'}}>Modify</Button>
          </div>
          </form>
          </div>
        <div style={{ height: '50vh', width: '80%',display:'flex',justifyContent:'center',alignItems:'center' }}>
      <DataGrid onRowClick={(event)=>handleclick(event)} rows={rows} columns={columns} pageSize={5} />
    </div>
    </div></div>
    )

}

export default Modifycourse