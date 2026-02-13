import { useState } from 'react';
import { View, Text, StyleSheet, FlatList, Alert, Platform } from 'react-native';
import { usePhotos } from '@/contexts/PhotoContext';
import { PhotoItem } from '@/components/PhotoItem';
import { Button } from '@/components/Button';
import { photoUploadService } from '@/services/photoUploadService';
import { router } from 'expo-router';
import { Image as ImageIcon } from 'lucide-react-native';

export default function GalleryScreen() {
  const { photos, removePhoto, clearPhotos } = usePhotos();
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (photos.length === 0) {
      if (Platform.OS !== 'web') {
        Alert.alert('Erreur', 'Veuillez prendre au moins une photo');
      }
      return;
    }

    setLoading(true);
    try {
      const imageUris = photos.map((photo) => photo.uri);
      const response = await photoUploadService.uploadPhotos(imageUris);

      router.push({
        pathname: '/results',
        params: { message: response.reponse.message },
      });
    } catch (error) {
      console.error('Upload error:', error);
      if (Platform.OS !== 'web') {
        Alert.alert('Erreur', "Impossible d\'envoyer les photos. Vérifiez votre connexion.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = (id: string) => {
    removePhoto(id);
  };

  const handleClearAll = () => {
    if (Platform.OS !== 'web') {
      Alert.alert(
        'Confirmer',
        'Voulez-vous vraiment supprimer toutes les photos ?',
        [
          { text: 'Annuler', style: 'cancel' },
          { text: 'Supprimer', onPress: clearPhotos, style: 'destructive' },
        ]
      );
    } else {
      clearPhotos();
    }
  };

  if (photos.length === 0) {
    return (
      <View style={styles.container}>
        <View style={styles.emptyState}>
          <ImageIcon size={64} color="#cbd5e1" strokeWidth={1.5} />
          <Text style={styles.emptyTitle}>Aucune photo</Text>
          <Text style={styles.emptyMessage}>
            Prenez des photos pour commencer l&apos;analyse
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Galerie</Text>
          <Text style={styles.subtitle}>{photos.length} photo{photos.length > 1 ? 's' : ''}</Text>
        </View>
        <Button
          title="Tout supprimer"
          onPress={handleClearAll}
          variant="secondary"
        />
      </View>

      <FlatList
        data={photos}
        renderItem={({ item }) => (
          <PhotoItem photo={item} onRemove={handleRemove} />
        )}
        keyExtractor={(item) => item.id}
        numColumns={2}
        columnWrapperStyle={styles.row}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
      />

      <View style={styles.footer}>
        <Button
          title="Analyser les photos"
          onPress={handleUpload}
          loading={loading}
          disabled={loading}
        />
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
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#0f172a',
  },
  subtitle: {
    fontSize: 14,
    color: '#64748b',
    marginTop: 4,
  },
  listContent: {
    paddingHorizontal: 24,
    paddingBottom: 20,
  },
  row: {
    justifyContent: 'space-between',
  },
  footer: {
    padding: 24,
    borderTopWidth: 1,
    borderTopColor: '#f1f5f9',
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 48,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#0f172a',
    marginTop: 24,
  },
  emptyMessage: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 24,
  },
});
