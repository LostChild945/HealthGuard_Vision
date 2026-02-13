import { ApiResponse } from '@/types/api';

class PhotoUploadService {
    private apiKey: string;
    private baseUrl: string;

    constructor() {
        this.apiKey = process.env.EXPO_PUBLIC_API_KEY || '';
        this.baseUrl = process.env.EXPO_PUBLIC_API_BASE_URL || '';
    }

    async uploadPhotos(imageUris: string[]): Promise<ApiResponse> {
        try {
            const formData = new FormData();

            imageUris.forEach((uri, index) => {
                const fileName = `photo_${index}.jpg`;
                formData.append('files', {
                    uri,
                    name: fileName,
                    type: 'image/jpeg',
                } as any);
            });

            const url = `${this.baseUrl.replace(/\/$/, '')}/upload?key-api=${encodeURIComponent(this.apiKey)}`;

            const response = await fetch(url, {
                method: 'POST',
                body: formData,
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