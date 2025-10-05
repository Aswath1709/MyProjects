import { useState } from 'react';


const Login = ({admin,student,faculty}) => {


    const [username, setUsername] = useState(admin?'Username':student?'Roll Number':'Email ID')
    const [placeholder, setPlaceholder] = useState(admin?'abc123':student?'CB.EN.U4CSE18001':'abc@xyz.com')
    const [inputType, setInputType] = useState(admin?'text':student?'text':'email')

   const submitHandler=(e)=>{
       e.preventDefault();
       if(admin){
        window.location='/admin'
       }

       if(student){
        window.location='/student'
       }

       if(faculty){
        window.location='/faculty'
       }

   }

    return ( 
        <div className="card login-form-card">
        <form onSubmit={submitHandler} className="admin-form">
            {admin===1&&<h4 className="form-header">Admin Login</h4>}
            {student===1&&<h4 className="form-header">Student Login</h4>}
            {faculty===1&&<h4 className="form-header">Faculty Login</h4>}
            <div className="form-group">
                <label className="login-form-label">{username}</label>
                <input type={inputType} className="form-control login-form-input" placeholder={placeholder}></input>
            </div>
            <div className="form-group">
                <label className="login-form-label">Password</label>
                <input type="password" className="form-control login-form-input" placeholder="Pass"></input>
            </div>

            <button type="submit" class="btn btn-primary">Login</button>
        </form>
        <a href="www.google.com">Forgo</a>
        </div>
     );
}
 
export default Login;