import '../index.css'
import Button from '@material-ui/core/Button';
import Container from '@material-ui/core/Container';
import ReactFontLoader from 'react-font-loader';
import { makeStyles } from '@material-ui/core/styles';
import {useSpring, animated} from 'react-spring'


const useStyles = makeStyles((theme) => ({
    root: {
      width: '100%',
    },
    rootContainer:{
       marginTop:'16%',
       textAlign:'center'
    },
    heading: {
      fontSize: theme.typography.pxToRem(15),
      fontWeight: theme.typography.fontWeightRegular,
    },
    login:{
      fontWeight:'bolder',
      marginBottom:'3%',
      width:'wrap',
      backgroundColor:'rgb(60,60,60)',
      color:'white',
      borderRadius:'7px',
      paddingTop:'6px',
      paddingBottom:'6px'
    }
  }));

const Email = ({setFunc}) => {
    const submitHandler=(e)=>{
        e.preventDefault();
        alert("Email successfully sent!");
        }
        const props = useSpring({
            config: { duration:250 },
            
            from: { transform: 'translate3d(0,-60px,0)' },
            to: { transform: 'translate3d(0,0px,0)' },
          })
        
          const props2 = useSpring({
            config: { duration:300 },
            
            from: { transform: 'translate3d(0,120px,0)',
                    opacity:0 },
            to: { transform: 'translate3d(0,0px,0)',
                  opacity:1 },
          })
        
          const props3 = useSpring({
            config: { duration:350 },
            
            from: { opacity : 0 },
            to: { opacity : 1 },
          })
          const classes = useStyles();        
    return ( 
        <animated.div style={props3} className="main">
        <ReactFontLoader url='https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap' />
    <Container className={classes.rootContainer} maxWidth="sm">
        <animated.h2 className={classes.login} style={props2}>Enter Email Address</animated.h2>
    <animated.div className={classes.root} style={props2}>

        <div>
         <form onSubmit={submitHandler} className="Add-form">
         <div className="form-group">
                <input type="text" className="email-form-input" placeholder="Email"></input>
            </div>
            <Button type="submit" class="btn btn-primary" variant="contained" color="primary" disableElevation style={{height:'100%', marginLeft:'2%', backgroundColor:'rgb(60,60,60)',fontFamily:'Kosugi Maru'}}>Submit</Button>
        </form>
         </div>
         </animated.div>
    </Container>
        </animated.div>
     );
}
 
export default Email;