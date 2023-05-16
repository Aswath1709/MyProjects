import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';
import AddCircleIcon from '@material-ui/icons/AddCircle';
import EditIcon from '@material-ui/icons/Edit';
import DeleteIcon from '@material-ui/icons/Delete';
import HomeIcon from '@material-ui/icons/Home';
import ExitToAppIcon from '@material-ui/icons/ExitToApp';
import '../fonts/KosugiMaru-Regular.ttf'
import EnhancedEncryptionIcon from '@material-ui/icons/EnhancedEncryption';
import DragHandleIcon from '@material-ui/icons/DragHandle';


const MenuAppBar= ({setFunc}) => {
    const useStyles = makeStyles((theme) => ({
        root: {
          flexGrow: 1,
        },
        menuButton: {
          marginRight: theme.spacing(2),
        },
        title: {
          flexGrow: 1,
        },
      
        typography: {
          fontFamily: [
            'KosugiMaru',
            'cursive',
          ].join(','),
        },
      }));
      
  const classes = useStyles();
  
 
    const handleMenuAdd=()=>{
      window.location='/admin/add'
        
    }

    const handleMenuAssign=()=>{
      window.location='/admin/assign'
        
    }

    const handleMenuEdit=()=>{
      window.location='/admin/modify'
    }

    const handleMenuDelete=()=>{
      window.location='/admin/delete'
    }

    const handleMenuHome=()=>{
      window.location='/admin'
    }
    const handleMenuExit=()=>{
      localStorage.removeItem('user');
      window.location='/logout'
    }
    const handleMenuChange=()=>{
      window.location='/email'
        
    }

  return (
    <div className={classes.root}>
      <AppBar style={{
          backgroundColor:'rgb(60,60,60)'}} position="static">
        <Toolbar>
          <Typography variant="h6" className={classes.title}>
           Attendance Manager
          </Typography>
            <div>
              <IconButton
                onClick={handleMenuAdd}
                color="inherit"
              >
                  <AddCircleIcon style={{color:"white"}}/>
                
              </IconButton>

              <IconButton
                onClick={handleMenuAssign}
                color="inherit"
              >
                  <DragHandleIcon style={{color:"white"}}/>
                
              </IconButton>
              
              <IconButton
                onClick={handleMenuEdit}
                color="inherit"
              >
                  <EditIcon style={{color:"white"}}/>
                
              </IconButton>
              
              <IconButton
                onClick={handleMenuDelete}
                color="inherit"
              >
                  <DeleteIcon style={{color:"white"}}/>
                
              </IconButton>
              <IconButton
                onClick={handleMenuHome}
                color="inherit"
              >
                  <HomeIcon style={{color:"white"}}/>
                
              </IconButton>
              {/* <IconButton
                onClick={handleMenuChange}
                color="inherit"
              >
                  <EnhancedEncryptionIcon style={{color:"white"}}/>
              </IconButton> */}

              <IconButton
                onClick={handleMenuExit}
                color="inherit"
              >
                  <ExitToAppIcon style={{color:"white"}}/>
              </IconButton>

            </div>
          
        </Toolbar>
      </AppBar>
    </div>
  );
}

export default MenuAppBar; 
