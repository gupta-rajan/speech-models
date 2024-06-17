// api.js (located in the root directory)
import axios from "axios";

const apiUrl = "http://10.195.250.59:8000";

const api = axios.create({
  baseURL: apiUrl,
});

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('audio', file);

  try {
    const token = localStorage.getItem('accessToken'); // Adjust this based on how your token is stored
    const response = await api.post('/speech/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${token}`, // Include the authorization header
      },
    });

    return response.data;
  } catch (error) {
    throw new Error(error.response.data.message || 'Error uploading file');
  }
};

export default api;