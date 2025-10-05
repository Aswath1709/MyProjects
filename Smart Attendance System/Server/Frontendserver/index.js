const express = require("express");
const app = express();


app.use(express.static('public'));

app.get('*',(req,res)=>{
	let path=require('path')
	res.sendFile(path.resolve('public/index.html'));
})

app.listen(3000)