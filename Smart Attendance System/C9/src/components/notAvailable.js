import Paper from '@material-ui/core/Paper';
import ReactFontLoader from 'react-font-loader';
import * as React from 'react';
import { CircularProgress } from '@material-ui/core';


const NotAuth = () => {

    const [disp,setDisp]=React.useState(0)

    React.useEffect(()=>{
        setTimeout(function() { 
            setDisp(1)
        }, 5000)
        setTimeout(function() { 
            setDisp(2)
        }, 10000)
    
    },[])
    
    return ( 
        <div style={{height:'90vh',display:'flex',justifyContent:'center',alignItems:'center'}}>
      <ReactFontLoader url='https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap' />
            
            <Paper style={{padding:'4vh 4vw',display:'flex',justifyContent:'center',alignItems:'center',flexDirection:'column'}}>
                {disp<2&&<CircularProgress/>}
                {disp===1&&<p style={{marginTop:'1vh',fontSize:'1em',fontFamily: 'Kosugi Maru',fontWeight:'bold'}}>This is taking too long. Please wait.</p>}              
                {disp===2&&<p style={{fontSize:'1.5em',fontFamily: 'Kosugi Maru',fontWeight:'bold'}}>Oops! Something's wrong :(</p>}
            </Paper>
        </div>
     );
}
 
export default NotAuth;