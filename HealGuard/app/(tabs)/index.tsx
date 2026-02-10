import { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, Platform } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Camera, FlipHorizontal, Image as ImageIcon } from 'lucide-react-native';
import { usePhotos } from '@/contexts/PhotoContext';
import { Button } from '@/components/Button';
import { router } from 'expo-router';

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const { addPhoto, photos } = usePhotos();

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Chargement de la caméra...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Camera size={64} color="#2563eb" strokeWidth={2} />
        <Text style={styles.title}>Accès à la caméra requis</Text>
        <Text style={styles.message}>
          Cette application a besoin d&apos;accéder à votre caméra pour prendre des photos
        </Text>
        <Button title="Autoriser l&apos;accès" onPress={requestPermission} />
      </View>
    );
  }

  const takePicture = async () => {
    if (!cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
      });

      if (photo?.uri) {
        addPhoto(photo.uri);
        if (Platform.OS !== 'web') {
          Alert.alert('Succès', 'Photo ajoutée à la galerie');
        }
      }
    } catch (error) {
      console.error('Error taking picture:', error);
      if (Platform.OS !== 'web') {
        Alert.alert('Erreur', 'Impossible de prendre la photo');
      }
    }
  };

  const toggleCameraFacing = () => {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  };

  const goToGallery = () => {
    router.push('/gallery');
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Prendre une photo</Text>
        {photos.length > 0 && (
          <TouchableOpacity style={styles.galleryButton} onPress={goToGallery}>
            <ImageIcon size={24} color="#2563eb" strokeWidth={2} />
            <View style={styles.badge}>
              <Text style={styles.badgeText}>{photos.length}</Text>
            </View>
          </TouchableOpacity>
        )}
      </View>

      <CameraView ref={cameraRef} style={styles.camera} facing={facing}>
        <View style={styles.cameraOverlay}>
          <View style={styles.frameGuide} />
        </View>
      </CameraView>

      <View style={styles.controls}>
        <TouchableOpacity style={styles.iconButton} onPress={toggleCameraFacing}>
          <FlipHorizontal size={28} color="#2563eb" strokeWidth={2} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.captureButton} onPress={takePicture}>
          <View style={styles.captureButtonInner} />
        </TouchableOpacity>

        <View style={styles.iconButton} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 20,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#0f172a',
  },
  galleryButton: {
    position: 'relative',
    padding: 8,
  },
  badge: {
    position: 'absolute',
    top: 0,
    right: 0,
    backgroundColor: '#ef4444',
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  badgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#0f172a',
    marginTop: 24,
    marginBottom: 12,
  },
  message: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
    marginBottom: 32,
    paddingHorizontal: 32,
    lineHeight: 24,
  },
  camera: {
    flex: 1,
    marginHorizontal: 24,
    marginBottom: 24,
    borderRadius: 24,
    overflow: 'hidden',
  },
  cameraOverlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  frameGuide: {
    width: '80%',
    aspectRatio: 1,
    borderWidth: 2,
    borderColor: '#fff',
    borderRadius: 16,
    opacity: 0.5,
  },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    paddingBottom: 40,
  },
  iconButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#f1f5f9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#fff',
  },
});
