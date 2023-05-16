import Navbar from './navbar';
import Button from '@material-ui/core/Button';
import * as React from 'react';
import { DataGrid, GRID_ROW_SELECTED } from '@material-ui/data-grid';
import DeleteIcon from '@material-ui/icons/Delete';
import { makeStyles } from '@material-ui/core/styles';
import TextField from '@material-ui/core/TextField';
import { useEffect } from 'react';



const CourseDelete = ({ Courses }) => {
    const columns = [
        { field: 'id', headerName: 'ID', width: 200 },
        { field: 'C_Name', headerName: 'Course Name', width: 200 },
        { field: 'C_Type', headerName: 'Username', width: 200 },
        {
            field: 'C_Sem',
            headerName: 'Semester',
            type: 'number',
            width: 200,
        },
    ];

    const [courses, setCourses] = React.useState(Courses);
    async function getcourses() {
        fetch("http://localhost:4000/courses")
            .then(response => response.json())
            .then(response => setCourses(response.data))
            .catch(err => console.error(err))
    }
    useEffect(() => {
        getcourses()
    }, [courses])
    const [rowind, setRowind] = React.useState(null)
    const rows = courses
    const handleclick = (e) => {
        setRowind(e.rowIndex)
    }

    const handleDelete = () => {
        console.log(rowind) 
        console.log(rows[rowind].id)       
        fetch(`http://localhost:4000/courses/delete?id=${rows[rowind].id}`)
        .then(response=>response.json())
        .catch(err=>console.error(err))   
    }

    return (
        <div>
            <Navbar />
            <h3 style={{ marginTop: '3vh', paddingLeft: '2vw', marginBottom: '3vh' }}><u>Delete Course</u></h3>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
                <div style={{ height: '50vh', width: '80%', display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '3vh' }}>
                    <DataGrid onRowClick={(event) => handleclick(event)} rows={rows} columns={columns} pageSize={5} />
                </div>
                <Button onClick={handleDelete} variant="contained"><DeleteIcon />Delete</Button>
            </div>
        </div>

    );
}

export default CourseDelete;