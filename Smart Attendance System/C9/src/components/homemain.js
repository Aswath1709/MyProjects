import { useState } from 'react';
import { AiFillDownCircle, AiFillUpCircle } from "react-icons/ai";
import Login from './login'
import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Accordion from '@material-ui/core/Accordion';
import AccordionSummary from '@material-ui/core/AccordionSummary';
import AccordionDetails from '@material-ui/core/AccordionDetails';
import Typography from '@material-ui/core/Typography';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import Container from '@material-ui/core/Container';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import ReactFontLoader from 'react-font-loader';
import { useSpring, animated } from 'react-spring'
import Link from '@material-ui/core/Link';
var CryptoJS = require("crypto-js");


const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
  },
  rootContainer: {
    marginTop: '16%',
    textAlign: 'center'
  },
  heading: {
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular,
  },
  login: {
    fontWeight: 'bolder',
    marginBottom: '3%',
    width: 'wrap',
    backgroundColor: 'rgb(60,60,60)',
    color: 'white',
    borderRadius: '7px',
    paddingTop: '6px',
    paddingBottom: '6px'
  }
}));

const Homemain = () => {


  const props = useSpring({
    config: { duration: 250 },

    from: { transform: 'translate3d(0,-60px,0)' },
    to: { transform: 'translate3d(0,0px,0)' },
  })

  const props2 = useSpring({
    config: { duration: 300 },

    from: {
      transform: 'translate3d(0,120px,0)',
      opacity: 0
    },
    to: {
      transform: 'translate3d(0,0px,0)',
      opacity: 1
    },
  })

  const props3 = useSpring({
    config: { duration: 350 },

    from: { opacity: 0 },
    to: { opacity: 1 },
  })


  const classes = useStyles();


  const [accord, setAccord] = useState(0)
  const [uname, setUname] = useState('')
  const [pass, setPass] = useState('')
  const [errMsg, setErrMsg] = useState('')

  const adminSubmitHandler = (e) => {
    e.preventDefault();
    if (uname === "admin" && pass === "admin") {
      setErrMsg("")
      var data = [{ uname: `${uname}`, pass: `${pass}` }]
          var ciphertext = CryptoJS.AES.encrypt(JSON.stringify(data), 'my-secret-key@123').toString();
          // console.log(ciphertext);
          localStorage.setItem('user', ciphertext);
      setAccord(0)
      window.location = '/admin'

    }
    else {
      setErrMsg("Invalid Credentials")
    }
  }

  const changeUname = (e) => {
    setUname(e.target.value)
  }

  const changePass = (e) => {
    setPass(e.target.value)
  }

  const studentSubmitHandler = (e) => {
    e.preventDefault();
    setAccord(0)
    fetch(`http://localhost:4000/students/login?uname=${uname}&pass=${pass}`)
      .then(response => response.json())
      .then(response => {
        if (response.data.length === 0) {
          setErrMsg('username does not exist')
          console.log('Username does not exist')
        }

        else if (response.data[0].S_Password !== pass) {
          setErrMsg('Password does not match')
          console.log('Password does not match')
        }

        else {
          console.log('User signed in')
          setErrMsg('')
          var data = [{ uname: `${uname}`, pass: `${pass}` }]
          var ciphertext = CryptoJS.AES.encrypt(JSON.stringify(data), 'my-secret-key@123').toString();
          // console.log(ciphertext);
          localStorage.setItem('user', ciphertext);
          window.location = `/student/${response.data[0].id}`

          // data=localStorage.getItem('user');
          // var bytes = CryptoJS.AES.decrypt(data, 'my-secret-key@123');
          // var decryptedData = JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
          // console.log(decryptedData)

  

        }



      })
      .catch(err => console.error(err))
  }



  const facultySubmitHandler = (e) => {
    e.preventDefault();
    setAccord(0)
    fetch(`http://localhost:4000/faculties/login?uname=${uname}&pass=${pass}`)
      .then(response => response.json())
      .then(response => {
        if (response.data.length === 0) {
          setErrMsg('username does not exist')
          console.log('Username does not exist')
        }

        else if (response.data[0].F_Password !== pass) {
          setErrMsg('Password does not match')
          console.log('Password does not match')
        }

        else {
          console.log('User signed in')
          setErrMsg('')
          var data = [{ uname: `${uname}`, pass: `${pass}` }]
          var ciphertext = CryptoJS.AES.encrypt(JSON.stringify(data), 'my-secret-key@123').toString();
          // console.log(ciphertext);
          localStorage.setItem('user', ciphertext);
          window.location = `/faculty/${response.data[0].id}`
        }



      })
      .catch(err => console.error(err))
  }

  const preventDefault = (e) => {
    e.preventDefault();
    setAccord(0)
    window.location = '/email'
  }

  return (
    <animated.div style={props3} className="main">
      <ReactFontLoader url='https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap' />
      <Container className={classes.rootContainer} maxWidth="sm">
        <animated.h2 className={classes.login} style={props2}>Login</animated.h2>
        <animated.div className={classes.root} style={props2}>
          {(accord === 0 || accord === 1) && (
            <Accordion >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="panel1a-content"
                id="panel1a-header"
                onClick={() => {
                  if (accord === 1) setAccord(0)
                  else setAccord(1)
                }}
              >
                <Typography style={{ fontFamily: 'Kosugi Maru' }} className={classes.heading}>Admin</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <form onSubmit={adminSubmitHandler} style={{ width: '100%' }}>
                  <TextField style={{ fontFamily: 'Kosugi Maru' }}
                    id="outlined-admin-username-input"
                    value={uname}
                    onChange={changeUname}
                    label="Username"
                    type="text"
                    variant="outlined"
                  />

                  <TextField style={{ marginLeft: '3%' }}
                    id="outlined-admin-password-input"
                    value={pass}
                    onChange={changePass}
                    label="Password"
                    type="password"
                    variant="outlined"
                  />
                  <div>
                    <p style={{ color: 'red' }}>
                      {errMsg}
                    </p>
                    <Button type='submit' variant="contained" color="primary" disableElevation style={{ height: '5vh', marginTop: '2vh', marginLeft: '0%', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>
                      Submit
        </Button>
                  </div>
                </form>
              </AccordionDetails>
            </Accordion>)}
          {(accord !== 0 && accord !== 1) && (
            <Accordion disabled style={{ backgroundColor: 'rgba(40, 40,40, 0.5)', color: 'white' }}>
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="panel3a-content"
                id="panel3a-header"
              >
                <Typography style={{ fontFamily: 'Kosugi Maru' }} className={classes.heading}>Admin</Typography>
              </AccordionSummary>
            </Accordion>)}
          {(accord === 0 || accord === 2) && (
            <Accordion >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="panel1a-content"
                id="panel1a-header"
                onClick={() => {
                  if (accord === 2) setAccord(0)
                  else setAccord(2)
                }}
              >
                <Typography style={{ fontFamily: 'Kosugi Maru' }} className={classes.heading}>Student</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <form onSubmit={studentSubmitHandler} style={{ width: '100%' }}>
                  <TextField style={{ fontFamily: 'Kosugi Maru' }}
                    id="outlined-student-username-input"
                    value={uname}
                    onChange={changeUname}
                    label="Username"
                    type="text"
                    variant="outlined"
                  />

                  <TextField style={{ marginLeft: '3%' }}
                    id="outlined-student-password-input"
                    value={pass}
                    onChange={changePass}
                    label="Password"
                    type="password"
                    variant="outlined"
                  />
                  <div>
                    <p style={{ color: 'red' }}>
                      {errMsg}
                    </p>
                    <Button type='submit' variant="contained" color="primary" disableElevation style={{ height: '5vh', marginTop: '2vh', marginLeft: '0%', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>
                      Submit
        </Button>
                  </div>
                </form>
              </AccordionDetails>
            </Accordion>)}
          {(accord !== 0 && accord !== 2) && (
            <Accordion disabled style={{ backgroundColor: 'rgba(40, 40,40, 0.5)', color: 'white' }}>
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="panel3a-content"
                id="panel3a-header"
              >
                <Typography style={{ fontFamily: 'Kosugi Maru' }} className={classes.heading}>Student</Typography>
              </AccordionSummary>
            </Accordion>)}
          {(accord === 0 || accord === 3) && (
            <Accordion >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="panel1a-content"
                id="panel1a-header"
                onClick={() => {
                  if (accord === 3) setAccord(0)
                  else setAccord(3)
                }}
              >
                <Typography style={{ fontFamily: 'Kosugi Maru' }} className={classes.heading}>Faculty</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <form onSubmit={facultySubmitHandler} style={{ width: '100%' }}>
                  <TextField style={{ fontFamily: 'Kosugi Maru' }}
                    id="outlined-faculty-username-input"
                    value={uname}
                    onChange={changeUname}
                    label="Username"
                    type="text"
                    variant="outlined"
                  />

                  <TextField style={{ marginLeft: '3%' }}
                    id="outlined-faculty-password-input"
                    value={pass}
                    onChange={changePass}
                    label="Password"
                    type="password"
                    variant="outlined"
                  />
                  <div>
                    <p style={{ color: 'red' }}>
                      {errMsg}
                    </p>
                    <Button type='submit' variant="contained" color="primary" disableElevation style={{ height: '5vh', marginTop: '2vh', marginLeft: '0%', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>
                      Submit
        </Button>
                  </div>
                </form>
              </AccordionDetails>
            </Accordion>)}
          {(accord !== 0 && accord !== 3) && (
            <Accordion disabled style={{ backgroundColor: 'rgba(40, 40,40, 0.5)', color: 'white' }}>
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="panel3a-content"
                id="panel3a-header"
              >
                <Typography style={{ fontFamily: 'Kosugi Maru' }} className={classes.heading}>Faculty</Typography>
              </AccordionSummary>
            </Accordion>)}
          {/* <Typography className={classes.root}>
            <Link href="#" style={{ fontFamily: 'Kosugi Maru' }} onClick={preventDefault}>Forgot Password</Link>
          </Typography> */}
        </animated.div>
      </Container>
    </animated.div>
  );
}

export default Homemain;