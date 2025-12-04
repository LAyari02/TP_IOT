import paho.mqtt.client as mqtt
import json
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

# Configuration MQTT
BROKER = "localhost"
PORT = 1883
TOPIC = "ecg/signal"

# Stockage des données reçues
temps_recu = []
valeurs_recues = []
reception_terminee = False

# Modèle LSTM
model = None
sequence_length = 50
signal_min = None
signal_max = None

def normaliser_signal(signal):
    """Normalise un signal avec les bornes apprises lors de l'entraînement."""
    if signal_min is None or signal_max is None:
        raise ValueError("Bornes de normalisation manquantes dans model_params.pkl")
    amplitude = signal_max - signal_min
    if np.isclose(amplitude, 0.0):
        raise ValueError("Amplitude de normalisation nulle")
    return (signal - signal_min) / amplitude

def denormaliser_signal(signal_norm):
    """Restaure l'échelle d'origine du signal normalisé."""
    if signal_min is None or signal_max is None:
        raise ValueError("Bornes de normalisation manquantes dans model_params.pkl")
    return signal_norm * (signal_max - signal_min) + signal_min

def charger_modele():
    """Charge le modèle LSTM pré-entraîné"""
    global model, sequence_length, signal_min, signal_max
    
    if not os.path.exists('ecg_lstm_model.h5'):
        print("❌ ERREUR: Modèle non trouvé!")
        print("   Veuillez d'abord exécuter lstm_training.py pour entraîner le modèle.")
        return False
    
    print("📥 Chargement du modèle LSTM...")
    try:
        model = load_model('ecg_lstm_model.h5', compile=False)
    except ValueError as exc:
        print(f"❌ Échec du chargement du modèle: {exc}")
        print("   Vérifiez que le fichier a été entraîné avec une version compatible de Keras.")
        return False
    
    # Charger les paramètres
    with open('model_params.pkl', 'rb') as f:
        params = pickle.load(f)
        sequence_length = params['sequence_length']
        signal_min = params.get('signal_min')
        signal_max = params.get('signal_max')
    if signal_min is None or signal_max is None:
        print("❌ Paramètres de normalisation absents dans model_params.pkl")
        print("   Réentraînez le modèle avec LSTM.py après mise à jour.")
        return False
    
    print(f"✅ Modèle chargé (sequence_length={sequence_length})")
    return True

def on_connect(client, userdata, flags, rc):
    """Callback appelé lors de la connexion"""
    if rc == 0:
        print("✅ Edge connecté au broker MQTT")
        client.subscribe(TOPIC)
        print(f"📡 Abonné au topic: {TOPIC}\n")
    else:
        print(f"❌ Échec de connexion, code: {rc}")

def on_message(client, userdata, msg):
    """Callback appelé à la réception d'un message"""
    global reception_terminee
    
    try:
        data = json.loads(msg.payload.decode())
        
        # Vérifier si c'est un message de fin
        if data.get("type") == "end":
            print("\n✅ Réception terminée!")
            reception_terminee = True
            return
        
        # Stocker les données
        temps_recu.append(data["timestamp"])
        valeurs_recues.append(data["value"])
        
        print(f"📥 Reçu: t={data['timestamp']:.3f}s, valeur={data['value']:.4f}")
        
    except Exception as e:
        print(f"❌ Erreur de traitement: {e}")

def generer_signal_original():
    """Génère le signal ECG original pour comparaison"""
    def ecg_synthetique(t):
        return (
            0.1 * np.sin(2 * np.pi * t * 1) +
            -0.15 * np.exp(-((t - 0.25) ** 2) / 0.001) +
            1.0 * np.exp(-((t - 0.3) ** 2) / 0.0005) +
            -0.2 * np.exp(-((t - 0.35) ** 2) / 0.001) +
            0.3 * np.exp(-((t - 0.5) ** 2) / 0.01)
        )
    
    t = np.linspace(0, 1, 500)
    cycle = ecg_synthetique(t)
    signal_complet = np.tile(cycle, 5)
    temps_total = np.linspace(0, 5, len(signal_complet))
    
    return temps_total, signal_complet

def reconstruire_avec_lstm(valeurs_echantillonnees):
    """
    Reconstruit le signal complet en utilisant le LSTM
    
    Args:
        valeurs_echantillonnees: points ECG échantillonnés reçus
    
    Returns:
        signal reconstruit
    """
    print("\n🧠 Reconstruction du signal avec LSTM...")
    
    valeurs_echantillonnees = np.asarray(valeurs_echantillonnees, dtype=np.float32)
    valeurs_norm = normaliser_signal(valeurs_echantillonnees)
    signal_reconstruit_norm = list(valeurs_norm[:sequence_length])
    
    # Prédire les points manquants
    for i in range(len(valeurs_echantillonnees) - sequence_length):
        # Prendre les derniers 'sequence_length' points
        sequence = np.array(signal_reconstruit_norm[-sequence_length:])
        sequence = sequence.reshape(1, sequence_length, 1)
        
        # Prédire le prochain point
        prediction = model.predict(sequence, verbose=0)[0, 0]
        signal_reconstruit_norm.append(prediction)
        
        # Tous les N points, on utilise la vraie valeur reçue pour recalibrer
        if (i + sequence_length) < len(valeurs_echantillonnees):
            # Remplacer par la vraie valeur pour éviter l'accumulation d'erreurs
            if i % 5 == 0:  # Recalibration tous les 5 points
                signal_reconstruit_norm[-1] = valeurs_norm[i + sequence_length]
    
    print(f"✅ Signal reconstruit: {len(signal_reconstruit_norm)} points")
    
    signal_reconstruit = denormaliser_signal(np.array(signal_reconstruit_norm))
    return signal_reconstruit

def interpoler_lineairement(temps, valeurs, nb_points_cible=2500):
    """Interpolation linéaire simple pour comparaison"""
    temps_interp = np.linspace(temps[0], temps[-1], nb_points_cible)
    valeurs_interp = np.interp(temps_interp, temps, valeurs)
    return temps_interp, valeurs_interp

def calculer_metriques(signal_original, signal_reconstruit):
    """Calcule les métriques de qualité"""
    # S'assurer que les signaux ont la même longueur
    min_len = min(len(signal_original), len(signal_reconstruit))
    sig_orig = signal_original[:min_len]
    sig_recon = signal_reconstruit[:min_len]
    
    mse = np.mean((sig_orig - sig_recon) ** 2)
    mae = np.mean(np.abs(sig_orig - sig_recon))
    rmse = np.sqrt(mse)
    
    # Corrélation
    correlation = np.corrcoef(sig_orig, sig_recon)[0, 1]
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation
    }

def afficher_resultats_avec_lstm():
    """Affiche les résultats avec reconstruction LSTM"""
    print("\n" + "=" * 70)
    print("   RECONSTRUCTION AVEC LSTM")
    print("=" * 70)
    
    # Générer le signal original
    temps_original, signal_original = generer_signal_original()
    
    # Reconstruire avec LSTM
    signal_lstm = reconstruire_avec_lstm(valeurs_recues)
    
    # Interpolation linéaire pour comparaison
    temps_interp, signal_interp = interpoler_lineairement(
        temps_recu, valeurs_recues, nb_points_cible=len(signal_original)
    )
    
    # Calculer les métriques
    print("\n📊 MÉTRIQUES DE QUALITÉ:")
    print("\n   LSTM vs Original:")
    temps_lstm = np.linspace(temps_recu[0], temps_recu[-1], len(signal_lstm))
    signal_lstm_interp = np.interp(temps_original, temps_lstm, signal_lstm)
    metriques_lstm = calculer_metriques(signal_original, signal_lstm_interp)
    print(f"      MSE:  {metriques_lstm['mse']:.6f}")
    print(f"      MAE:  {metriques_lstm['mae']:.6f}")
    print(f"      RMSE: {metriques_lstm['rmse']:.6f}")
    print(f"      Corrélation: {metriques_lstm['correlation']:.4f}")
    
    print("\n   Interpolation Linéaire vs Original:")
    metriques_interp = calculer_metriques(signal_original, signal_interp)
    print(f"      MSE:  {metriques_interp['mse']:.6f}")
    print(f"      MAE:  {metriques_interp['mae']:.6f}")
    print(f"      RMSE: {metriques_interp['rmse']:.6f}")
    print(f"      Corrélation: {metriques_interp['correlation']:.4f}")
    
    # Amélioration
    amelioration = ((metriques_interp['rmse'] - metriques_lstm['rmse']) / 
                    metriques_interp['rmse'] * 100)
    print(f"\n🎯 Amélioration LSTM vs Interpolation: {amelioration:.1f}%")
    
    # Visualisation
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # 1. Signal original
    axes[0].plot(temps_original, signal_original, 'b-', linewidth=1.5, label='Signal Original')
    axes[0].set_title('1. Signal ECG Original (Haute résolution)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Amplitude (mV)', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Signal échantillonné (reçu)
    axes[1].plot(temps_recu, valeurs_recues, 'ro-', linewidth=2, 
                markersize=6, label=f'Points Reçus ({len(valeurs_recues)} points)')
    axes[1].set_title(f'2. Signal Échantillonné Reçu via MQTT (~{len(valeurs_recues)//5} points/cycle)', 
                     fontsize=13, fontweight='bold', color='darkred')
    axes[1].set_ylabel('Amplitude (mV)', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Reconstruction par interpolation linéaire
    axes[2].plot(temps_interp, signal_interp, 'orange', linewidth=1.5, 
                label='Interpolation Linéaire', alpha=0.8)
    axes[2].plot(temps_recu, valeurs_recues, 'ro', markersize=4, label='Points Reçus')
    axes[2].set_title('3. Reconstruction par Interpolation Linéaire', 
                     fontsize=13, fontweight='bold', color='darkorange')
    axes[2].set_ylabel('Amplitude (mV)', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Reconstruction LSTM
    axes[3].plot(temps_original, signal_lstm_interp, 'g-', linewidth=1.5, 
                label='Reconstruction LSTM', alpha=0.8)
    axes[3].plot(temps_recu, valeurs_recues, 'ro', markersize=4, label='Points Reçus')
    axes[3].set_title(f'4. Reconstruction par LSTM (Amélioration: {amelioration:.1f}%)', 
                     fontsize=13, fontweight='bold', color='darkgreen')
    axes[3].set_xlabel('Temps (s)', fontsize=11)
    axes[3].set_ylabel('Amplitude (mV)', fontsize=11)
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'ecg_reconstruction_lstm_{len(valeurs_recues)}_points.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Graphique sauvegardé: {filename}")
    
    plt.show()

def recevoir_et_reconstruire():
    """Fonction principale"""
    global reception_terminee
    
    print("=" * 70)
    print("   EDGE IoT - RÉCEPTEUR ECG AVEC RECONSTRUCTION LSTM")
    print("=" * 70)
    
    # Charger le modèle LSTM
    if not charger_modele():
        return
    
    client = mqtt.Client("ECG_Edge_LSTM")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        print("\n🔌 Connexion au broker MQTT...\n")
        
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        
        # Attendre la fin de la réception
        print("⏳ En attente des données...\n")
        while not reception_terminee:
            pass
        
        # Reconstruire et afficher les résultats
        if len(valeurs_recues) > sequence_length:
            afficher_resultats_avec_lstm()
        else:
            print(f"❌ Pas assez de données reçues (minimum: {sequence_length} points)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("\n👋 Déconnexion du broker")

if __name__ == "__main__":
    recevoir_et_reconstruire()