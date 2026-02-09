import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { useLocalSearchParams, router } from 'expo-router';
import { Button } from '@/components/Button';
import { CheckCircle } from 'lucide-react-native';
import { usePhotos } from '@/contexts/PhotoContext';

export default function ResultsScreen() {
  const params = useLocalSearchParams<{ message: string }>();
  const { clearPhotos } = usePhotos();

  const handleNewAnalysis = () => {
    clearPhotos();
    router.replace('/');
  };

  const handleViewGallery = () => {
    router.back();
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <View style={styles.iconContainer}>
          <CheckCircle size={64} color="#10b981" strokeWidth={2} />
        </View>
        <Text style={styles.title}>Analyse terminée</Text>
        <Text style={styles.subtitle}>Voici les résultats de l'analyse</Text>
      </View>

      <View style={styles.resultCard}>
        <Text style={styles.resultLabel}>Résultat</Text>
        <Text style={styles.resultText}>{params.message || 'Aucun message reçu'}</Text>
      </View>

      <View style={styles.actions}>
        <Button
          title="Nouvelle analyse"
          onPress={handleNewAnalysis}
        />
        <View style={styles.spacing} />
        <Button
          title="Retour à la galerie"
          onPress={handleViewGallery}
          variant="secondary"
        />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  content: {
    padding: 24,
    paddingTop: 80,
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
  },
  iconContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#d1fae5',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
  },
  resultCard: {
    backgroundColor: '#f8fafc',
    borderRadius: 16,
    padding: 24,
    marginBottom: 40,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  resultLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 12,
  },
  resultText: {
    fontSize: 18,
    color: '#0f172a',
    lineHeight: 28,
  },
  actions: {
    gap: 12,
  },
  spacing: {
    height: 12,
  },
});
