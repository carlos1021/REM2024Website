const functions = require('firebase-functions');
const admin = require('firebase-admin');
const express = require('express');
const multer = require('multer');
const path = require('path');

// Initialize Firebase Admin SDK
admin.initializeApp();

// Create an Express appconst app = express();

// Set up Multer for file uploadsconst upload = multer({
    storage: multer.diskStorage({
        destination: function (req, file, cb) {
            cb(null, 'uploads/'); // Set the destination for uploaded files
        },
        filename: function (req, file, cb) {
            cb(null, Date.now() + path.extname(file.originalname)); // Set the filename
        }
});

// Middleware to handle multipart/form-data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Route to handle file uploads
app.post('/upload', upload.single('file'), (req, res) => {
    if (req.file) {
        console.log('File received:', req.file);
        res.send(req.file.filename);
    } else {
        res.status(400).send('No file uploaded.');
    }
});

// Export the Express app as a Firebase Functionexports.api = functions.https.onRequest(app);
