import NavBar from "./user-navbar";
import ReactFontLoader from 'react-font-loader';
import { useParams } from 'react-router-dom'
import * as React from 'react'
import Downloader from 'react-base64-downloader'
import CircularProgress from '@material-ui/core/CircularProgress';
import NotAuth from './notAvailable';
var CryptoJS = require("crypto-js");


// const msgs = [
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'Lorem sit incididunt sint fugiat consectetur cupidatat Lorem ad duis consequat ad velit et. Tempor nostrud eiusmod in ea magna magna culpa. Ullamco nisi eu laborum cillum qui eiusmod occaecat esse dolor ex mollit laborum. Deserunt cupidatat esse fugiat fugiat labore laborum. Et consectetur ad minim fugiat anim nulla dolor ex.'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'Excepteur voluptate incididunt aliqua pariatur esse Lorem cupidatat amet do. Nostrud enim ea mollit proident ad commodo eu ex duis amet. Magna tempor nulla ut est anim pariatur mollit. Deserunt in irure mollit officia excepteur et sit culpa. Officia id ex do voluptate officia nisi et consequat incididunt voluptate nostrud do cillum. Quis pariatur amet ipsum irure qui velit in laboris anim non qui consequat do ipsum. Mollit ea deserunt et incididunt tempor ipsum consequat sint nostrud nulla.'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'Enim proident laborum est et dolor non. Dolor pariatur magna sit duis dolor aliqua. Non elit laborum reprehenderit nulla.'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'Adipisicing nulla officia elit Lorem commodo qui. Enim sint eu laborum cillum aute qui ex. Adipisicing do esse sint aliqua non.'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'Reprehenderit cupidatat consectetur sint id sunt. Fugiat elit labore aliquip dolor labore laborum reprehenderit reprehenderit nulla ex tempor. Dolore officia occaecat exercitation consectetur ut et ea id laboris ea mollit duis.'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'Labore tempor nostrud et quis laboris consequat occaecat. Sint exercitation anim aliquip aliquip non id cupidatat do cillum officia. Qui proident fugiat esse nisi consequat culpa ad sunt incididunt Lorem laborum occaecat proident ipsum. In occaecat eu culpa anim. Quis sunt nostrud dolor ipsum dolor cupidatat eu consequat ea tempor sunt exercitation tempor.'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'Consectetur aliqua consectetur officia ex. Ad voluptate cupidatat sunt eiusmod voluptate amet esse. Eu qui consectetur aliquip veniam amet voluptate. Velit nulla cillum consectetur mollit aliquip laborum nulla velit laboris laborum sit.'
//     },
// ]

const Facultymsg = () => {

    var { id } = useParams()
    const [msgs, setMsgs] = React.useState(null)
    const [local,setLocal]=React.useState(localStorage.getItem('user')||null)
    const [auth,setAuth]=React.useState(0)

    async function getstudents() {
        fetch(`http://localhost:4000/msgstudent/retrieve?fid=${id}`)
            .then(response => response.json())
            .then(response => setMsgs(response.data))
            .catch(err => console.error(err))
    }
    React.useEffect(() => {
        getstudents()
        async function authfn() {
            if(local){
            const response = await fetch(`http://localhost:4000/faculties/auth?id=${id}`)
            const json = await response.json()
            // console.log(json.data[0])
            var bytes = CryptoJS.AES.decrypt(local, 'my-secret-key@123');
            var decr = JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
            // console.log(decr[0].pass===json.data[0].S_Password)
            var v=decr[0].uname===json.data[0].F_Username&&decr[0].pass===json.data[0].F_Password?1:0
            setAuth(v)}
    
    
      
          }
        authfn()

    }, [])

    const handleAck=(mid)=>{
        console.log(mid)
        fetch(`http://localhost:4000/msg/ack?fid=${mid}`)
            .catch(err => console.error(err))
            window.location.reload()
    }

    return (
        <div>{auth===0?(<div>
            <NotAuth/>
          </div>):
        (<div>
            <ReactFontLoader url='https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap' />
            <NavBar message={"faculty"} fid={id}/>
            <h4 style={{ marginLeft: '4vw', marginTop: '5vh' }}><u>MESSAGE HISTORY</u></h4>
            <div>
            {!msgs&&(<div style={{display: 'flex', flexDirection: 'column', width: '100%', alignItems: 'center', justifyContent: 'center',marginTop:'10vh'}}><CircularProgress color="black" /></div>)}
                {msgs&&(<div style={{ marginTop: '2vh', display: 'flex', flexDirection: 'column', width: '100%', alignItems: 'center', justifyContent: 'flex-start', height: '90%', overflowY: 'scroll' }}>
                    {
                        msgs.map((m, index) => (
                            <div style={{ width: '90vw', marginTop: '2vh', borderBottom: '1px solid grey', paddingBottom: '1.5vh' }}>
                                <div style={{ display: 'grid', width: '100%', fontFamily: 'Kosugi Maru', gridTemplateColumns: '1fr 1fr', fontWeight: 'bold' }}>
                                    <div style={{display:'flex',flexDirection:'row',justifyContent:'flex-start',alignItems:'center'}}>
                                        <h6 style={{ fontWeight: 'bolder' }}>
                                            {`From: ${m.SId}`}
                                        </h6>
                                        <p style={{background:m.Ack?'green':'red',padding:'0.25vw 0.5vw',color:'white',marginLeft:'0.75vw',borderRadius:'75px'}}>Ack</p>
                                    </div>
                                    <div>
                                    <div style={{ width: '100%', textAlign: 'right', fontFamily: 'Kosugi Maru' }}>
                                            <h6 style={{ fontWeight: 'bolder' }}>{`Sent: ${m.Dt}`}</h6>
                                        </div>
                                    </div>
                                    <h6 style={{ fontWeight: 'bolder' }}>
                                        {`${m.Msg}`}
                                    </h6>

                                </div>
                                <div style={{display: 'grid', width: '100%', fontFamily: 'Kosugi Maru', gridTemplateColumns: '1fr 1fr'}}>
                                {m.Doc!==null&&(<Downloader style={{height: '5vh', marginTop:'1vh', width: 'fit-content', color: 'white', border: '0', borderRadius: '4px', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru'}} base64={m.Doc}>Download Document</Downloader>)}
                                {m.Doc===null&&(<p style={{color:'white'}}>.</p>)}
                                <div style={{textAlign:'right'}}>
                                {m.Ack===0&&(<button style={{height: '5vh', width: 'fit-content', color: 'white', border: '0', borderRadius: '4px', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru'}} onClick={()=>handleAck(m.MsgId)} >Acknowledge</button>)}
                                </div>
                            </div>
                            </div>
                        ))
                    }
                </div>)}
            </div>
        </div>)}
        </div>
    );
}

export default Facultymsg;