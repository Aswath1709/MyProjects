import '../index.css'
import Navbar from './user-navbar';
import * as React from 'react'
import { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import ReactFontLoader from 'react-font-loader';
import Link from '@material-ui/core/Link';
import { blueGrey } from '@material-ui/core/colors';
import  Chart  from "react-google-charts";
import { useParams } from 'react-router-dom'
import NotAuth from './notAvailable';
var CryptoJS = require("crypto-js");


const StudentHome = () => {
  var { id } = useParams()
  const [name, setName] = React.useState('')
  const [data, setData] = React.useState([['Course', 'Avg(Attendance)']])
  const [local,setLocal]=React.useState(localStorage.getItem('user')||null)
  const [auth,setAuth]=React.useState(0)


  React.useEffect(() => {
    fetch(`http://localhost:4000/student/home?sid=${id}`)
      .then(response => response.json())
      .then(response => setName(response.data))
      .catch(err => console.error(err))

      async function authfn() {
        if(local){
        const response = await fetch(`http://localhost:4000/students/auth?id=${id}`)
        const json = await response.json()
        console.log(json.data[0])
        var bytes = CryptoJS.AES.decrypt(local, 'my-secret-key@123');
        var decr = JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
        console.log(decr[0].pass===json.data[0].S_Password)
        var v=decr[0].uname===json.data[0].S_Username&&decr[0].pass===json.data[0].S_Password?1:0
        setAuth(v)}


  
      }

      async function fn() {
      const response = await fetch(`http://localhost:4000/student/home/graph?fid=${id}`)
      const json = await response.json()
      const arr = [['Course', 'Avg(Attendance)',{ role: 'style' }]]
      for (var i = 0; i < json.data.length; i++) {
        var color=(json.data[i].av<75?'#ff0000':'green')
        arr.push([json.data[i].CId, json.data[i].av,color])
      }
      setData(arr)

    }
    authfn()

    fn()

  }, [])

  const preventDefault = (e) => {
    e.preventDefault();
    window.location = `/attendance/${id}`
  }

  const preventDefault1 = (e) => {
    e.preventDefault();
    window.location = `/student/${id}/msg`
  }

  const useStyles = makeStyles((theme) => ({
    root: {
      flexGrow: 1,
      marginTop: '10%',
    },
    paper: {
      padding: theme.spacing(2),
      textAlign: 'center',
      color: theme.palette.text.secondary,
      height: '100%',
    },

    titleNumber: {
      backgroundColor: 'rgb(180, 180, 180)'

    }
  }));


  const classes = useStyles();

  const [functionality, setFunctionality] = useState('home');

  const functionalityHandler = (arg) => {
    setFunctionality(arg)
  }
  return (
    <div>{
      auth===0?(<div>
        <NotAuth/>
      </div>):
    (<div>
      <ReactFontLoader url='https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap' />
      <Navbar message="student" fid={id} />
      <Grid container spacing={3}>
        <Grid style={{display:'flex',justifyContent:'center',alignItems:'center',marginTop:'3vh'}} item xs={12} >
          <Chart
             width={'80vw'}
             height={'50vh'}
            chartType="ColumnChart"
            loader={<div>Loading Chart</div>}
            data={data}
            options={{
              // Material design options
              chart: {
                title: 'Attendance Summary',

              },
              // colors:['green'],
            }}
            // For tests
            rootProps={{ 'data-testid': '2' }}
          />
        </Grid>
      </Grid>

      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center',marginTop:'4vh' }}>
        <Grid style={{ width: '85vw' }} container spacing={10}>
          <Grid item xs={12} spacing={24} >
            <Paper className={classes.paper} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Typography variant="h3" style={{ fontFamily: 'Kosugi Maru', color: 'black', fontWeight: 'bolder', fontSize: '2em' }} className={classes.title}>
                {name && (`Welcome ${name[0].S_Name} :)`)}
              </Typography>
            </Paper>
          </Grid>

          <Grid style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }} item xs={4} spacing={24} >
            <Paper className={classes.paper} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '20vw', background: 'rgb(60,60,60)' }}>
              <Link href="https://intranet.cb.amrita.edu/TimeTable/" style={{ fontFamily: 'Kosugi Maru' }} ><Typography variant="h3" style={{ fontFamily: 'Kosugi Maru', color: 'white', fontWeight: 'bolder', fontSize: '2em' }} className={classes.title}>View my Timetable</Typography></Link>

            </Paper>
          </Grid>

          <Grid style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }} item xs={4} spacing={24} >
            <Paper className={classes.paper} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '20vw', background: 'rgb(60,60,60)' }}>

              <Link href="#" style={{ fontFamily: 'Kosugi Maru' }} onClick={preventDefault}><Typography variant="h3" style={{ fontFamily: 'Kosugi Maru', color: 'white', fontWeight: 'bolder', fontSize: '2em' }} className={classes.title}>View my Attendance</Typography></Link>

            </Paper>
          </Grid>

          <Grid style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }} item xs={4} spacing={24} >
            <Paper className={classes.paper} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '20vw', background: 'rgb(60,60,60)' }}>

              <Link href="" onClick={preventDefault1}><Typography variant="h3" style={{ fontFamily: 'Kosugi Maru', color: 'white', fontWeight: 'bolder', fontSize: '2em' }} className={classes.title}>Messages</Typography></Link>

            </Paper>
          </Grid>

          {/* <Grid item xs={6}>
        <Paper className={classes.paper}>
            <Typography variant="h3" style={{ fontFamily: 'Kosugi Maru', color: 'black',  marginBottom: '7%',fontWeight:'bolder',fontSize:'250%' }} className={classes.title}>
            <Link href="https://intranet.cb.amrita.edu/TimeTable/" style={{fontFamily:'Kosugi Maru'}} >View my Timetable</Link>
            </Typography>
            <Typography variant="h3" style={{ fontFamily: 'Kosugi Maru', color: 'black', marginTop: '7%',fontWeight:'bolder',fontSize:'250%' }} className={classes.title}>
            <Link href="#" style={{fontFamily:'Kosugi Maru'}} onClick={preventDefault}>View my Attendance</Link>
            </Typography>
            <Typography variant="h3" style={{ fontFamily: 'Kosugi Maru', color: 'black', marginTop: '7%',fontWeight:'bolder',fontSize:'250%' }} className={classes.title}>
            <Link href="" style={{fontFamily:'Kosugi Maru'}} onClick={preventDefault1}>Messages</Link>
            </Typography>
        </Paper>
      </Grid> */}

        </Grid>
      </div>
    </div>)}
    </div>
  );
}

export default StudentHome;