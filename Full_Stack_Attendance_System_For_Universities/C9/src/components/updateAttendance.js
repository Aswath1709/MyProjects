import NavBar from "./user-navbar";
import ReactFontLoader from 'react-font-loader';
import * as React from 'react'
import TextField from '@material-ui/core/TextField';
import { useParams } from 'react-router-dom'
import NotAuth from './notAvailable';
import background1 from '../assets/AttendBackground1.png'
import background2 from '../assets/AttendBackground2.png'
import background3 from '../assets/AttendBackground3.png'


var CryptoJS = require("crypto-js");

const UpdateAttendance = () => {

    var { id } = useParams()
    const [sid, setSId] = React.useState(null)
    const [cid, setCId] = React.useState(null)
    const [date, setDate] = React.useState(null)
    const [local, setLocal] = React.useState(localStorage.getItem('user') || null)
    const [auth, setAuth] = React.useState(0)

    React.useEffect(() => {

        async function authfn() {
            if (local) {
                const response = await fetch(`http://localhost:4000/faculties/auth?id=${id}`)
                const json = await response.json()
                // console.log(json.data[0])
                var bytes = CryptoJS.AES.decrypt(local, 'my-secret-key@123');
                var decr = JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
                // console.log(decr[0].pass===json.data[0].S_Password)
                var v = decr[0].uname === json.data[0].F_Username && decr[0].pass === json.data[0].F_Password ? 1 : 0
                setAuth(v)
            }



        }

        authfn()
    }, [])

    const onDateChange = (e) => {
        setDate(e.target.value)
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        fetch(`http://localhost:4000/updateattend/od?sid=${sid}&fid=${id}&cid=${cid}&date=${date}`)
            .then(response => response.json())
            .catch(err => console.error(err))

        fetch(`http://localhost:4000/updateattendance/od?sid=${sid}&fid=${id}&cid=${cid}`)
            .then(response => response.json())
            .catch(err => console.error(err))
    }

    return (
        <div>{auth === 0 ? (<div>
            <NotAuth />
        </div>) :
            (<div>
                <ReactFontLoader url='https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap' />
                <NavBar message={"faculty"} fid={id} />
                <div style={{
                    background: `url(${background3})`,
                    height: '91.5vh',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat',
                    backgroundSize: 'cover',
                }}>
                    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', height: '90vh', alignItems: 'center' }}>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <label>Student ID</label>
                            <input required style={{ marginLeft: '0', width: '12vw' }} type="text" id="" className="form-attendance" placeholder="SId" value={sid} onChange={(e) => setSId(e.target.value)}></input>
                        </div>
                        <div style={{ marginTop: '1vh' }}>
                            <label >Course ID</label><br></br>
                            <input required style={{ marginLeft: '0', width: '12vw' }} type="text" id="" className="form-attendance" placeholder="CId" value={cid} onChange={(e) => setCId(e.target.value)}></input>
                        </div>
                        <div style={{ marginTop: '1vh' }}>
                            <TextField
                                id="date"
                                label="Date"
                                type="date"
                                required
                                value={date}
                                onChange={onDateChange}
                                InputLabelProps={{
                                    shrink: true,
                                }}
                            />
                        </div>
                        <button type='submit' style={{ marginTop: '3vh', height: '5vh', width: '5vw', backgroundColor: 'rgb(60,60,60)', fontFamily: 'Kosugi Maru', border: '0', borderRadius: '4px', color: 'white' }}>Update</button>
                    </form>
                </div>
            </div>)}
        </div>

    );
}

export default UpdateAttendance;