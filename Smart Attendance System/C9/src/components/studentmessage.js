import NavBar from "./user-navbar";
import ReactFontLoader from 'react-font-loader';
import { useParams } from 'react-router-dom'
import * as React from 'react'
import Downloader from 'react-base64-downloader'
import CircularProgress from '@material-ui/core/CircularProgress';
import NotAuth from './notAvailable';
var CryptoJS = require("crypto-js");

const fileToDataUri = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
        resolve(event.target.result)
    };
    reader.readAsDataURL(file);
})
// const msgs = [
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'hi hello how r u'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'hi hello how r u'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'hi hello how r u'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'hi hello how r u'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'hi hello how r u'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'hi hello how r u'
//     },
//     {
//         'to': 'aswanth',
//         'date': '01/06/2021',
//         'msg': 'hi hello how r u'
//     },
// ]

const Studentmsg = () => {
    var { id } = useParams()
    const [msgs, setMsgs] = React.useState(null)
    const [faculty, setFaculty] = React.useState(null)
    const [msg, setMsg] = React.useState(null)
    const [fid, setFid] = React.useState(null)
    const [date,setDate]=React.useState((new Date()).getFullYear()+'-'+'0'+((new Date()).getMonth()+1)+'-'+(new Date()).getDate());
    const [local,setLocal]=React.useState(localStorage.getItem('user')||null)
    const [auth,setAuth]=React.useState(0)
    const [doc, setDoc] = React.useState(null)

    

    const submitHandler = (e) => {
        e.preventDefault()

        fetch(`http://localhost:4000/msg/post`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sid: id, msg: msg, doc: doc, fid: fid,date:date })
            })
            .then(()=>(window.location.reload(true)))
    }

    const handleChangeMsg = (e) => {
        setMsg(e.target.value)
    }

    const handleChangeFid = (e) => {
        setFid(e.target.value)
    }

    const handleChangeDoc = (file) => {
        // setDoc(e.target.files[0])
        if (!file) {
            setDoc('');
            return;
        }

        fileToDataUri(file)
            .then(dataUri => {
                setDoc(dataUri)
            })
    }

    async function getstudents() {
        fetch(`http://localhost:4000/student/retrievemsg?sid=${id}`)
            .then(response => response.json())
            .then(response => setMsgs(response.data))
            .catch(err => console.error(err))
        fetch(`http://localhost:4000/msgfaculty/retrieve?sid=${id}`)
            .then(response => response.json())
            .then(response => setFaculty(response.data))
            .catch(err => console.error(err))
    }
    React.useEffect(() => {
        getstudents()
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
        authfn()
        
    }, [])

    return (
        <div>{auth===0?(<div>
            <NotAuth/>
          </div>):
        (<div>
            <ReactFontLoader url='https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap' />
            <NavBar message={"student"} fid={id} />
            <h4 style={{ marginLeft: '4vw', marginTop: '5vh' }}><u>MESSAGE HISTORY</u></h4>
            {!msgs&&(<div style={{display: 'flex', flexDirection: 'column', width: '100%', alignItems: 'center', justifyContent: 'center',marginTop:'10vh'}}><CircularProgress color="black" /></div>)}
            {msgs && (
                <div style={{ marginTop: '2vh', display: 'flex', flexDirection: 'column', width: '100%', alignItems: 'center', justifyContent: 'flex-start', height: '40vh', overflowY: 'scroll' }}>
                    {

                        msgs.map((m, index) => (
                            <div>
                                <div style={{ width: '90vw', marginTop: '2vh', borderBottom: '1px solid grey', paddingBottom: '1.5vh' }}>
                                    <div style={{ display: 'grid', width: '100%', gridTemplateColumns: '1fr 1fr', fontFamily: 'Kosugi Maru', fontWeight: 'bold' }}>
                                        <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'flex-start', alignItems: 'center' }}>
                                            <h6 style={{ fontWeight: 'bolder' }}>
                                                {`To: ${m.F_Name}`}
                                            </h6>
                                            <p style={{ background: m.Ack ? 'green' : 'red', padding: '0.25vw 0.5vw', color: 'white', marginLeft: '0.75vw', borderRadius: '75px' }}>Ack</p>

                                        </div>
                                        <div style={{ width: '100%', textAlign: 'right', fontFamily: 'Kosugi Maru' }}>
                                            <h6 style={{ fontWeight: 'bolder' }}>{`Sent: ${m.Dt}`}</h6>
                                        </div>
                                    </div>
                                    <div >
                                        <h6 style={{ fontFamily: 'Kosugi Maru', color: 'navy' }}>
                                            {`${m.Msg}`}
                                        </h6>
                                    </div>
                                    {m.Doc !== null && (<Downloader style={{height: '5vh', width: 'fit-content', color: 'white', border: '0', borderRadius: '4px', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru'}} base64={m.Doc}>Download Document</Downloader>)}
                                </div>
                            </div>))
                    }
                    {/* <div style={{ width: '90vw', marginTop: '2vh' }}>
                    <div style={{ display: 'grid', width: '100%', gridTemplateColumns: '1fr 1fr', fontFamily: 'Kosugi Maru', fontWeight: 'bolder' }}>
                        <div>
                            <h6 style={{ fontWeight: 'bolder' }}>
                                To: Manoj
                            </h6>
                        </div>
                        <div style={{ width: '100%', textAlign: 'right', fontFamily: 'Kosugi Maru', fontWeight: 'bolder' }}>
                            <h6 style={{ fontWeight: 'bolder' }}>Sent : 21/10/2020</h6>
                        </div>
                    </div>
                    <div>
                        <h6 style={{ fontFamily: 'Kosugi Maru', color: 'navy' }}>
                            Commodo sint voluptate occaecat enim deserunt laborum quis adipisicing ea consectetur ipsum dolore. Nostrud nulla excepteur proident anim aliqua magna velit qui duis. Sint consequat ut qui minim exercitation dolor eu voluptate exercitation amet irure nulla do. Esse dolore irure consequat duis exercitation eiusmod aliqua dolor. Ex irure exercitation incididunt nisi labore duis cillum. Et et consectetur eiusmod ipsum.
                        </h6>
                    </div>
                </div> */}
                </div>)}
                {console.log(fid)}
            <div style={{ marginTop: '5vh', marginLeft: '4vw' }}>
                <h4><u>SEND MESSAGE</u></h4>
                <div style={{ display: 'flex', marginTop: '2vh', flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
                    {faculty && (<div>
                        <label className="login-form-label">Faculty Id</label>
                        <select value={fid} onChange={handleChangeFid} style={{ width: '10vw' }} id="Course" className="form-attendance">
                            <option value="0">Select Faculty</option>
                            {faculty.map((fac) => (<option value={`${fac.FId}`}>{fac.FId}</option>))}
                            {/* <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                            <option value="6">6</option>
                            <option value="7">7</option>
                            <option value="8">8</option> */}
                        </select>
                    </div>)}
                    <div style={{ height: '100%' }}>
                        <label className="login-form-label">Message</label>
                        <textarea value={msg} onChange={handleChangeMsg} style={{ width: '35vw', marginLeft: '3vw' }} rows='6' cols='50'></textarea>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center' }}>
                        <input onChange={(event) => handleChangeDoc(event.target.files[0] || null)} type='file'></input>
                        {console.log(typeof doc)}
                        <button onClick={submitHandler} style={{ height: '6vh', width: '14vw', color: 'white', border: '0', borderRadius: '4px', marginTop: '4%', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru' }}>Send</button>
                    </div>
                </div>
            </div>
        </div>)}
        </div>
    );
}

export default Studentmsg;