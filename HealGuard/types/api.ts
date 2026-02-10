export interface UploadPayload {
  key: string;
  imgs: string[];
}

export interface ApiResponse {
  reponse: {
    message: string;
  };
}

export interface Photo {
  id: string;
  uri: string;
  timestamp: number;
}
