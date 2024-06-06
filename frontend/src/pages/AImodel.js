import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadedFile(null); // Reset the uploaded file state on new selection
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first');
      return;
    }
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('audio', selectedFile);
      
      const response = await axios.post('http://127.0.0.1:8000/speech/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setUploadedFile(response.data.message);
    } catch (error) {
      console.error('Error uploading file: ', error);
      setUploadedFile('error');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-white">
      <div className="p-4 max-w-lg bg-white rounded-lg shadow-lg">
        <h1 className="text-2xl font-bold text-center mb-6 text-blue-700">Let's detect the Audio File</h1>
        
        <input
          type="file"
          accept=".wav"
          onChange={handleFileChange}
          className="block w-full text-lg text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none mb-4"
        />
        
        <button
          onClick={handleUpload}
          className="w-full bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white font-bold py-2 px-4 rounded transition duration-300"
          disabled={isUploading}
        >
          {isUploading ? 'Uploading...' : 'Upload'}
        </button>
        
        {isUploading && (
          <div className="w-full bg-gray-200 rounded-full h-4 mt-4">
            <div className="bg-blue-600 h-4 rounded-full animate-pulse" style={{ width: '100%' }}></div>
          </div>
        )}

        <div className="mt-3 text-center">
          {uploadedFile === 'error' && <p className="text-red-500">Error uploading file. Please try again.</p>}
          {uploadedFile === 0 && <p className="text-red-500 font-semibold">The voice input is Fake</p>}
          {uploadedFile === 1 && <p className="text-green-500 font-semibold">The voice input is Real</p>}
          {uploadedFile === null && <p className="text-gray-500">Please upload an audio file.</p>}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;