import Navbar from './navbar';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import AddCircleIcon from '@material-ui/icons/AddCircle';
import EditRoundedIcon from '@material-ui/icons/EditRounded';
import background1 from '../assets/AttendBackground1.png'
import background2 from '../assets/AttendBackground2.png'
import background3 from '../assets/AttendBackground3.png'
import background5 from '../assets/AttendBackground5.png'

const AdminModify = () => {
   
    const courseSubmitHandler=(e)=>{
        window.location='/admin/modify/course'
        }
    const facultySubmitHandler=(e)=>{
        window.location='/admin/modify/faculty'
        }
    const studentSubmitHandler=(e)=>{
        window.location="/admin/modify/student"
        }

    return (
    <div>
     <Navbar />
     <div style={{
                background: `url(${background5})`,
                height: '91.5vh',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                backgroundSize: 'cover',
            }}>
         <h4 className="form-header" disableElevation style={{paddingTop:'2%', marginBottom:'2%',marginRight:'2%', fontFamily:'Kosugi Maru'}}>Select</h4>
         <div className="form-group">
         <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'45%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Modify Course</Button>
         <IconButton
                onClick={courseSubmitHandler}
                color="inherit"
              >
                  <EditRoundedIcon style={{color:"black"}}/>
                  </IconButton>
            </div>
            <div className="form-group">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'45%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Modify Faculty</Button>
            <IconButton onClick={facultySubmitHandler} color="inherit">
                <EditRoundedIcon style={{color:"black"}}/>
            </IconButton>
            </div>
            <div className="form-group">
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'45%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Modify Student</Button>
                <IconButton onClick={studentSubmitHandler} color="inherit" >
                  <EditRoundedIcon style={{color:"black"}}/>
                </IconButton>
            </div>
            </div>
    </div>
      );
}
 


export default AdminModify;