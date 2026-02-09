import { View, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { X } from 'lucide-react-native';
import { Photo } from '@/types/api';

interface PhotoItemProps {
  photo: Photo;
  onRemove: (id: string) => void;
}

export function PhotoItem({ photo, onRemove }: PhotoItemProps) {
  return (
    <View style={styles.container}>
      <Image source={{ uri: photo.uri }} style={styles.image} />
      <TouchableOpacity
        style={styles.removeButton}
        onPress={() => onRemove(photo.id)}
      >
        <X size={20} color="#fff" strokeWidth={2} />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '48%',
    aspectRatio: 1,
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 16,
    backgroundColor: '#f1f5f9',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  removeButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: '#ef4444',
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
