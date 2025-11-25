// capture.js
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');


// Ask for camera permission
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
video.srcObject = stream;
video.play();
}).catch(function(err){
console.log('Camera permission denied or not available', err);
});
}


captureBtn.addEventListener('click', function(){
const ctx = canvas.getContext('2d');
ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
const dataURL = canvas.toDataURL('image/jpeg');


// Create a form and POST to /upload
const form = document.createElement('form');
form.method = 'POST';
form.action = '/upload';


const input = document.createElement('input');
input.type = 'hidden';
input.name = 'image';
input.value = dataURL;
form.appendChild(input);


document.body.appendChild(form);
form.submit();
});