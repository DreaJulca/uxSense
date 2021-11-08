const express = require('express'); // main library for server-client routing
const fs = require('fs'); // file system
//const path = require('path'); // what do i need this for?
const multer = require('multer'); // file storing middleware
const bodyParser = require('body-parser'); //cleans our req.body

const app = express();

/**
 * handle body requests, account for JSON parsing
 */
app.use(bodyParser.urlencoded({extended:false})); 
app.use(bodyParser.json());

/**
 * MULTER CONFIG: to get file videos to temp server storage
 */
const multerConfig = {
    
  storage: multer.diskStorage({ //Setup where the user's file will go
    destination: function(req, file, next){
      next(null, './uploads');
    },   
      
    //Then give the file a unique name
    filename: function(req, file, next){
        console.log(file);
        const ext = file.mimetype.split('/')[1];
        next(null, file.fieldname + '-' + Date.now() + '.'+ext);
    }
  }),   
  
  //A means of ensuring only videos are uploaded. 
  fileFilter: function(req, file, next){
        if(!file){
          next();
        }
      const video = file.mimetype.startsWith('video/');
      if(video){
        console.log('video uploaded');
        next(null, true);
      } else {
        console.log("file not supported");
        
        //TODO:  A better message response to user on failure.
        return next();
      }
  }
};

/**
   * Function to handle video playback and modeling
*/
function videoHandler(req, res) {
  const path = 'assets/' + req.query.vid + '.mp4'
  
  var spawn = require("child_process").spawn; 
  
	// Parameters passed in spawn - 
	// 1. type_of_script 
	// 2. list containing Path of the script 
	//    and arguments for the script  
	  
	// E.g : http://localhost:3000/name?firstname=Mike&lastname=Will 
  // so, first name = Mike and last name = Will 
  if(req.query.vid.substring(req.query.vid.length-6, req.query.vid.length-1) != "/poses"){
    var process = spawn('python', [
      "-u",
      __dirname + '/models/processing_manager.py',
     "--vid", path,
    ]); 
    // Takes stdout data from script which executed 
    // with arguments and send this data to res object
    // Actually, do not need to do this, and may cause crash. 
    /*
    process.stdout.on('data', function(data) { 
        res.write(data.toString()); 
    })
    */
  }
  
  const stat = fs.statSync(path)
  const fileSize = stat.size
  const range = req.headers.range

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-")
    const start = parseInt(parts[0], 10)
    const end = parts[1]
      ? parseInt(parts[1], 10)
      : fileSize-1

    const chunksize = (end-start)+1
    const file = fs.createReadStream(path, {start, end})
    const head = {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunksize,
      'Content-Type': 'video/mp4',
    }

    res.writeHead(206, head)
    file.pipe(res)
  } else {
    const head = {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
    }
    res.writeHead(200, head)
    fs.createReadStream(path).pipe(res)
  }
}


/**
   * Start server to host files for client
*/
app.use(express.static(__dirname + '/public'))

app.get('/', function(req, res) {
  res.sendFile(__dirname + '/index.htm')
})

/**
 * Handle video requests
 */

app.use('/video', videoHandler)

/**
 * Handle user uploads
 */
app.post('/upload',multer(multerConfig).single('vidupload'),function(req,res){
  res.send('Complete!');
});


app.listen(3000, function () {
  console.log('Listening on port 3000!')
});
