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
    { field: 'S_Name', headerName: 'Student Name', width: 130 },
    { field: 'S_Username', headerName: 'Username', width: 130 },
    { field: 'S_Password', headerName: 'Password', width: 130 },
    { field: 'C_elective', headerName: 'Elective', width: 130 },
    { field: 'S_Dept', headerName: 'Department', width: 130 },
    { field: 'S_Sec', headerName: 'Section', width: 130 },
    {
      field: 'S_Sem',
      headerName: 'Sem',
      type: 'number',
      width: 90,
    },
  ];



  
//      {Students.map((Student,index)=>{
//          {id:Student.S_Id, Name:Student.S_Name}
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

const Modifystudent = ({ Students }) => {
    const classes = useStyles();
    async function getstudents(){
      fetch("http://localhost:4000/students")
      .then(response=>response.json())
      .then(response=>setStudents(response.data))
      .catch(err=>console.error(err))
    }
    useEffect(()=>{
    getstudents()
})
    

    const handleclick=(e)=>{
      setRowind(e.rowIndex)
      setName(e.row.S_Name)
      setRno(e.row.id)
      setUname(e.row.S_Username)
      setPass(e.row.S_Password)
      setElective(e.row.C_elective)
      setDept(e.row.S_Dept)
      setSec(e.row.S_Sec)
      setSem(e.row.S_Sem)

    }

    const [name, setName] = React.useState('');
    const [students, setStudents] = React.useState(Students);
    const [rno, setRno] = React.useState('');
    const [uname, setUname] = React.useState('');
    const [pass, setPass] = React.useState('');
    const [elective, setElective] = React.useState(null);
    const [dept, setDept] = React.useState('');
    const [sem, setSem] = React.useState('');
    const [section, setSec] = React.useState('');
    const [rowind,setRowind]=React.useState(null)

    const rows = students
  
    const handleChangeName = (event) => {
    setName(event.target.value);
  };

  const handleChangeRno = (event) => {
    setRno(event.target.value);
  };

  const handleChangeUname = (event) => {
    setUname(event.target.value);
  };

  const handleChangePass = (event) => {
    setPass(event.target.value);
  };

  const handleChangeElective = (event) => {
    setElective(event.target.value);
  };

  const handleChangeDept = (event) => {
    setDept(event.target.value);
  };

  const handleChangeSec = (event) => {
    setSec(event.target.value);
  };

  const handleChangeSem = (event) => {
    setSem(event.target.value);
  };

  const submitHandler=(event)=>{
    
      //  e.preventDefault();
      // alert("Successfully added!");
      fetch(`http://localhost:4000/students/modify?id=${rno}&name=${name}&username=${uname}&password=${pass}&elective=${elective}&department=${dept}&section=${section}&semester=${sem}`)
      .then(response=>response.json())
      .catch(err=>console.error(err))

  //   // setName(e.row.S_Name)
  //   // setRno(e.row.id)
  //   // setUname(e.row.S_Username)
  //   // setPass(e.row.S_Password)
  //   // setElective(e.row.C_elective)
  //   // setDept(e.row.S_Dept)
  //   // setSem(e.row.S_Sem)
  }
    
    
    return (
      <div>
        <Navbar/>
        <h3 style={{marginTop:'3vh',paddingLeft:'2vw'}}><u>Modify Student</u></h3>
        <div style={{display:'flex',justifyContent:'center',alignItems:'center',flexDirection:'column'}}>
          <div style={{display:'flex',justifyContent:'center',alignItems:'center',marginTop:'3vh',marginBottom:'3vh'}}>
          <form onSubmit={submitHandler}>
          <TextField label="Roll No" disabled value={rno} onChange={handleChangeRno} style={{marginLeft:'2vh'}} />
          <TextField label="Name" value={name} onChange={handleChangeName} style={{marginLeft:'2vh'}}/>
          <TextField label="Username" value={uname} onChange={handleChangeUname} style={{marginLeft:'2vh'}}/>
          <TextField label="Password" value={pass} onChange={handleChangePass} style={{marginLeft:'2vh'}}/>
          <TextField label="Elective" value={elective} onChange={handleChangeElective} style={{marginLeft:'2vh'}}/>
          <TextField label="Department" value={dept} onChange={handleChangeDept} style={{marginLeft:'2vh'}}/>
          <TextField label="Section" value={section} onChange={handleChangeSec} style={{marginLeft:'2vh'}}/>
          <TextField label="Semester" value={sem} onChange={handleChangeSem} style={{marginLeft:'2vh'}}/>
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

export default Modifystudent