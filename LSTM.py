import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Fonction pour générer une onde ECG simulée
def ecg_synthetique(t):
    """Génère un signal ECG synthétique"""
    return (
        0.1 * np.sin(2 * np.pi * t * 1) +  # Onde P
        -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +  # Onde Q
        1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +  # Pic R
        -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +  # Onde S
        0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)  # Onde T
    )

def generer_donnees_entrainement(nb_cycles=50, points_par_cycle=500):
    """
    Génère des données d'entraînement avec variations
    
    Args:
        nb_cycles: nombre de cycles à générer
        points_par_cycle: résolution du signal
    
    Returns:
        signal complet pour l'entraînement
    """
    print(f"📊 Génération de {nb_cycles} cycles ECG pour l'entraînement...")
    
    t = np.linspace(0, 1, points_par_cycle)
    cycles = []
    
    for i in range(nb_cycles):
        # Générer le cycle de base
        cycle = ecg_synthetique(t)
        
        # Ajouter des variations pour rendre le modèle plus robuste
        # Variation d'amplitude (±10%)
        amplitude_var = np.random.uniform(0.9, 1.1)
        cycle = cycle * amplitude_var
        
        # Ajout de bruit
        bruit = 0.03 * np.random.normal(size=cycle.shape)
        cycle = cycle + bruit
        
        cycles.append(cycle)
    
    signal = np.concatenate(cycles)
    print(f"✅ Signal généré: {len(signal)} points")
    
    return signal

def normaliser_signal(signal):
    """Normalise le signal sur [0, 1] et retourne les bornes."""
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    if np.isclose(max_val, min_val):
        raise ValueError("Amplitude du signal nulle, impossible de normaliser")
    signal_norm = (signal - min_val) / (max_val - min_val)
    return signal_norm, min_val, max_val

def denormaliser_signal(signal_norm, min_val, max_val):
    """Restaure l'échelle originale d'un signal normalisé."""
    return signal_norm * (max_val - min_val) + min_val

def preparer_sequences(signal, sequence_length=50):
    """
    Prépare les séquences pour l'apprentissage LSTM
    
    Args:
        signal: signal ECG complet
        sequence_length: longueur de la séquence d'entrée
    
    Returns:
        X: séquences d'entrée
        y: valeurs à prédire
    """
    print(f"\n🔄 Préparation des séquences (longueur={sequence_length})...")
    
    X, y = [], []
    
    for i in range(len(signal) - sequence_length):
        X.append(signal[i:i + sequence_length])
        y.append(signal[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape pour LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print(f"✅ Forme X: {X.shape}")
    print(f"✅ Forme y: {y.shape}")
    
    return X, y

def creer_modele_lstm(sequence_length=50):
    """
    Crée l'architecture du modèle LSTM
    
    Args:
        sequence_length: longueur des séquences d'entrée
    
    Returns:
        modèle LSTM compilé
    """
    print("\n🏗️  Construction du modèle LSTM...")
    
    model = Sequential([
        # Première couche LSTM avec 128 unités
        LSTM(128, input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.2),  # Évite le surapprentissage
        
        # Deuxième couche LSTM avec 64 unités
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        # Couche Dense pour la prédiction
        Dense(32, activation='relu'),
        Dense(1)  # Sortie: prédiction d'un seul point
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("\n📋 Architecture du modèle:")
    model.summary()
    
    return model

def entrainer_modele(model, X_train, y_train, epochs=50, batch_size=64):
    """
    Entraîne le modèle LSTM
    
    Args:
        model: modèle à entraîner
        X_train, y_train: données d'entraînement
        epochs: nombre d'époques
        batch_size: taille des batches
    
    Returns:
        historique de l'entraînement
    """
    print("\n🚀 Début de l'entraînement...\n")
    
    # Callback pour arrêter si le modèle ne s'améliore plus
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("\n✅ Entraînement terminé!")
    
    return history

def visualiser_entrainement(history):
    """Visualise les courbes d'apprentissage"""
    plt.figure(figsize=(12, 4))
    
    # Perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss (entraînement)')
    plt.plot(history.history['val_loss'], label='Loss (validation)')
    plt.title('Évolution de la perte')
    plt.xlabel('Époque')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Erreur absolue moyenne
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE (entraînement)')
    plt.plot(history.history['val_mae'], label='MAE (validation)')
    plt.title('Erreur absolue moyenne')
    plt.xlabel('Époque')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
    print("\n💾 Graphique sauvegardé: lstm_training_history.png")
    plt.show()

def tester_modele(model, X_test, y_test, min_val, max_val):
    """Teste le modèle et visualise les prédictions"""
    print("\n🧪 Test du modèle...")
    
    y_pred = model.predict(X_test, verbose=0)
    
    # Revenir à l'échelle d'origine pour analyser la qualité réelle
    y_pred_denorm = denormaliser_signal(y_pred.flatten(), min_val, max_val)
    y_test_denorm = denormaliser_signal(y_test, min_val, max_val)
    
    mse = np.mean((y_test_denorm - y_pred_denorm) ** 2)
    mae = np.mean(np.abs(y_test_denorm - y_pred_denorm))
    
    print(f"📊 MSE: {mse:.6f}")
    print(f"📊 MAE: {mae:.6f}")
    
    # Visualisation
    plt.figure(figsize=(14, 5))
    
    nb_points = 1000
    plt.plot(y_test_denorm[:nb_points], label='Signal réel', linewidth=2)
    plt.plot(y_pred_denorm[:nb_points], label='Prédiction LSTM', 
             linestyle='--', linewidth=2, alpha=0.8)
    plt.title('Prédiction LSTM vs Signal Réel', fontsize=14, fontweight='bold')
    plt.xlabel('Index temporel')
    plt.ylabel('Amplitude (mV)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_prediction_test.png', dpi=300, bbox_inches='tight')
    print("💾 Graphique sauvegardé: lstm_prediction_test.png")
    plt.show()

def sauvegarder_modele(model, sequence_length, min_val, max_val):
    """Sauvegarde le modèle entraîné"""
    model.save('ecg_lstm_model.h5')
    
    # Sauvegarder aussi les paramètres
    params = {
        'sequence_length': sequence_length,
        'signal_min': min_val,
        'signal_max': max_val,
    }
    with open('model_params.pkl', 'wb') as f:
        pickle.dump(params, f)
    
    print("\n💾 Modèle sauvegardé:")
    print("   - ecg_lstm_model.h5")
    print("   - model_params.pkl")

def main():
    """Fonction principale"""
    print("=" * 70)
    print("   ENTRAÎNEMENT DU MODÈLE LSTM POUR RECONSTRUCTION ECG")
    print("=" * 70)
    
    # Paramètres
    NB_CYCLES = 50  # Nombre de cycles pour l'entraînement
    POINTS_PAR_CYCLE = 500
    SEQUENCE_LENGTH = 50  # Longueur de la séquence pour prédire le point suivant
    EPOCHS = 30
    BATCH_SIZE = 64
    
    # 1. Générer les données d'entraînement
    signal = generer_donnees_entrainement(NB_CYCLES, POINTS_PAR_CYCLE)
    signal_norm, signal_min, signal_max = normaliser_signal(signal)
    
    # 2. Préparer les séquences
    X, y = preparer_sequences(signal_norm, SEQUENCE_LENGTH)
    
    # 3. Diviser en train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Données d'entraînement: {len(X_train)} séquences")
    print(f"📊 Données de test: {len(X_test)} séquences")
    
    # 4. Créer le modèle
    model = creer_modele_lstm(SEQUENCE_LENGTH)
    
    # 5. Entraîner le modèle
    history = entrainer_modele(model, X_train, y_train, EPOCHS, BATCH_SIZE)
    
    # 6. Visualiser l'entraînement
    visualiser_entrainement(history)
    
    # 7. Tester le modèle
    tester_modele(model, X_test, y_test, signal_min, signal_max)
    
    # 8. Sauvegarder le modèle
    sauvegarder_modele(model, SEQUENCE_LENGTH, signal_min, signal_max)
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print("\nVous pouvez maintenant utiliser le modèle pour reconstruire")
    print("les signaux ECG échantillonnés reçus via MQTT.")

if __name__ == "__main__":
    main()