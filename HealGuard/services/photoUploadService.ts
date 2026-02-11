import { UploadPayload, ApiResponse } from '@/types/api';

class PhotoUploadService {
  private apiKey: string;
  private baseUrl: string;

  constructor() {
    this.apiKey = process.env.EXPO_PUBLIC_API_KEY || '';
    this.baseUrl = process.env.EXPO_PUBLIC_API_BASE_URL || '';
  }

  async uploadPhotos(imageUris: string[]): Promise<ApiResponse> {
    try {
      const payload: UploadPayload = {
        key: this.apiKey,
        imgs: imageUris,
      };

      const response = await fetch(`${this.baseUrl}/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      return data;
    } catch (error) {
      console.error('Error uploading photos:', error);
      throw error;
    }
  }
}

export const photoUploadService = new PhotoUploadService();
