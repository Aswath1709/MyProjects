import Navbar from './navbar';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import DeleteIcon from '@material-ui/icons/Delete';
import EditRoundedIcon from '@material-ui/icons/EditRounded';
import background1 from '../assets/AttendBackground1.png'
import background2 from '../assets/AttendBackground2.png'
import background3 from '../assets/AttendBackground3.png'
import background5 from '../assets/AttendBackground5.png'

const AdminDelete = () => {
   
    const courseSubmitHandler=(e)=>{
        window.location='/admin/delete/course'
        }
    const facultySubmitHandler=(e)=>{
        window.location='/admin/delete/faculty'
        }
    const studentSubmitHandler=(e)=>{
        window.location="/admin/delete/student"
        }

    return (
    <div>
     <Navbar />
        <div style={{
                background: `url(${background3})`,
                height: '91.5vh',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                backgroundSize: 'cover',
            }}>
         <h4 className="form-header" disableElevation style={{paddingTop:'2%', marginBottom:'2%',marginRight:'2%', fontFamily:'Kosugi Maru'}}>Select</h4>
         <div className="form-group">
         <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'45%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Delete Course</Button>
         <IconButton
                onClick={courseSubmitHandler}
                color="inherit"
              >
                  <DeleteIcon style={{color:"black"}}/>
                  </IconButton>
            </div>
            <div className="form-group">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'45%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Delete Faculty</Button>
            <IconButton onClick={facultySubmitHandler} color="inherit">
                <DeleteIcon style={{color:"black"}}/>
            </IconButton>
            </div>
            <div className="form-group">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'45%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Delete Student</Button>
                <IconButton onClick={studentSubmitHandler} color="inherit" >
                  <DeleteIcon style={{color:"black"}}/>
                </IconButton>
            </div>
            </div>
    </div>
      );
}
 


export default AdminDelete;