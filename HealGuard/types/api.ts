export interface UploadPayload {
  files: string[];
}

export interface ApiResponse {
    result: string;
}

export interface Photo {
  id: string;
  uri: string;
  timestamp: number;
}
