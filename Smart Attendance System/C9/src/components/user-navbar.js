import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';
import HomeIcon from '@material-ui/icons/Home';
import ExitToAppIcon from '@material-ui/icons/ExitToApp';
import '../fonts/KosugiMaru-Regular.ttf'
import EnhancedEncryptionIcon from '@material-ui/icons/EnhancedEncryption';
import EmailIcon from '@material-ui/icons/Email';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';

const NavBar = ({ message, fid }) => {
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

  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const classes = useStyles();

  const [user, setUser] = React.useState(message)
  const [Id, setId] = React.useState(fid)
  const [mail, setMail] = React.useState("")

  {
    console.log(mail)
    console.log(fid)
  }

  const handleMenuHome = () => {
    window.location = '/' + message + '/' + fid
  }
  const handleMenuExit = () => {
    localStorage.removeItem('user');
    window.location = '/logout'
  }
  const handleMenuChange = () => {
    async function fn(){
    await fetch(`http://localhost:4000/${message}/modifymail?id=${fid}&mail=${mail}`)
      .then(response => response.json())
      .catch(err => console.error(err))
    handleClose()  
  }
  fn()
}

  return (
    <div className={classes.root}>
      <AppBar style={{
        backgroundColor: 'rgb(60,60,60)'
      }} position="static">
        <Toolbar>
          <Typography variant="h6" className={classes.title}>
            Attendance Manager
          </Typography>
          <div>
            <IconButton
              onClick={handleMenuHome}
              color="inherit"
            >
              <HomeIcon style={{ color: "white" }} />

            </IconButton>

            <IconButton
              onClick={handleClickOpen}
              color="inherit"
            >
              <EmailIcon style={{ color: "white" }} />
            </IconButton>

            <IconButton
              onClick={handleMenuExit}
              color="inherit"
            >
              <ExitToAppIcon style={{ color: "white" }} />
            </IconButton>

          </div>

        </Toolbar>
      </AppBar>
      <Dialog open={open} onClose={handleClose} aria-labelledby="form-dialog-title">
        <div style={{ width: '30vw' }}>
          <DialogTitle id="form-dialog-title">Change E-Mail</DialogTitle>
          <DialogContent>
            <DialogContentText>
              Enter new Email
            </DialogContentText>
            <TextField
              autoFocus
              margin="dense"
              id="name"
              label="Email Address"
              type="email"
              fullWidth
              value={mail}
              onChange={e => setMail(e.target.value)}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleClose} color="primary">
              Cancel
            </Button>
            <Button onClick={handleMenuChange} color="primary">
              Done
            </Button>
          </DialogActions></div>
      </Dialog>
    </div>
  );
}

export default NavBar;
