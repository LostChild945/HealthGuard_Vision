import { createContext, useContext, useState, ReactNode } from 'react';
import { Photo } from '@/types/api';

interface PhotoContextType {
  photos: Photo[];
  addPhoto: (uri: string) => void;
  removePhoto: (id: string) => void;
  clearPhotos: () => void;
}

const PhotoContext = createContext<PhotoContextType | undefined>(undefined);

export function PhotoProvider({ children }: { children: ReactNode }) {
  const [photos, setPhotos] = useState<Photo[]>([]);

  const addPhoto = (uri: string) => {
    const newPhoto: Photo = {
      id: Date.now().toString(),
      uri,
      timestamp: Date.now(),
    };
    setPhotos((prev) => [...prev, newPhoto]);
  };

  const removePhoto = (id: string) => {
    setPhotos((prev) => prev.filter((photo) => photo.id !== id));
  };

  const clearPhotos = () => {
    setPhotos([]);
  };

  return (
    <PhotoContext.Provider value={{ photos, addPhoto, removePhoto, clearPhotos }}>
      {children}
    </PhotoContext.Provider>
  );
}

export function usePhotos() {
  const context = useContext(PhotoContext);
  if (context === undefined) {
    throw new Error('usePhotos must be used within a PhotoProvider');
  }
  return context;
}
