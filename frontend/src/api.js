import axios from "axios";

const apiUrl = "/choreo-apis/speechmodels/backend/v1";

const api = axios.create({
  baseURL: apiUrl,
});

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('audio', file);

  const response = await api.post('/speech/upload/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export default api;
