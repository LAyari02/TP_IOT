import numpy as np
import paho.mqtt.client as mqtt
import time
import json

# Configuration MQTT
BROKER = "localhost"  # ou l'adresse IP de votre broker
PORT = 1883
TOPIC = "ecg/signal"

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

def generer_signal_ecg(duree=5, points_par_cycle=500, nb_cycles=5):
    """
    Génère un signal ECG complet avec du bruit
    
    Args:
        duree: durée totale en secondes
        points_par_cycle: nombre de points par cycle cardiaque
        nb_cycles: nombre de cycles cardiaques
    """
    # Génération d'un cycle
    t = np.linspace(0, 1, points_par_cycle)
    cycle = ecg_synthetique(t)
    
    # Ajout de bruit pour simuler l'imprécision du capteur
    bruit = 0.03 * np.random.normal(size=cycle.shape)
    cycle_bruite = cycle + bruit
    
    # Répétition pour plusieurs battements
    signal_complet = np.tile(cycle_bruite, nb_cycles)
    temps_total = np.linspace(0, duree, len(signal_complet))
    
    return temps_total, signal_complet

def echantillonner_signal(temps, signal, nb_points_echantillon):
    """
    Échantillonne le signal en prenant seulement nb_points par cycle
    
    Args:
        temps: array des temps
        signal: array du signal
        nb_points_echantillon: nombre de points à garder par cycle
    """
    points_par_cycle = len(signal) // 5  # 5 cycles
    indices = []
    
    for i in range(5):  # Pour chaque cycle
        debut = i * points_par_cycle
        fin = (i + 1) * points_par_cycle
        # Prendre nb_points uniformément espacés dans chaque cycle
        indices_cycle = np.linspace(debut, fin-1, nb_points_echantillon, dtype=int)
        indices.extend(indices_cycle)
    
    return temps[indices], signal[indices]

def on_connect(client, userdata, flags, rc):
    """Callback appelé lors de la connexion au broker"""
    if rc == 0:
        print("✅ Connecté au broker MQTT")
    else:
        print(f"❌ Échec de connexion, code: {rc}")

def envoyer_signal_mqtt(nb_points_echantillon=20):
    """
    Génère et envoie le signal ECG via MQTT
    
    Args:
        nb_points_echantillon: nombre de points par cycle (5 ou 20)
    """
    # Créer le client MQTT
    client = mqtt.Client("ECG_Sensor")
    client.on_connect = on_connect
    
    try:
        # Connexion au broker
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        time.sleep(1)  # Attendre la connexion
        
        print(f"\n🫀 Génération du signal ECG...")
        print(f"📊 Échantillonnage: {nb_points_echantillon} points par cycle\n")
        
        # Générer le signal complet
        temps_complet, signal_complet = generer_signal_ecg()
        
        # Échantillonner le signal
        temps_echant, signal_echant = echantillonner_signal(
            temps_complet, signal_complet, nb_points_echantillon
        )
        
        # Envoyer les données échantillonnées
        print(f"📡 Envoi de {len(signal_echant)} points via MQTT...\n")
        
        for i, (t, valeur) in enumerate(zip(temps_echant, signal_echant)):
            message = {
                "timestamp": float(t),
                "value": float(valeur),
                "index": i
            }
            
            client.publish(TOPIC, json.dumps(message))
            print(f"Envoyé: t={t:.3f}s, valeur={valeur:.4f}")
            
            # Simuler un délai de transmission réaliste
            time.sleep(0.05)  # 50ms entre chaque envoi
        
        print("\n✅ Transmission terminée!")
        
        # Envoyer un message de fin
        client.publish(TOPIC, json.dumps({"type": "end"}))
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    print("=" * 60)
    print("   CAPTEUR ECG - SIMULATEUR IoT")
    print("=" * 60)
    
    # Choisir le taux d'échantillonnage
    print("\nChoisissez le taux d'échantillonnage:")
    print("1. 5 points par cycle (faible)")
    print("2. 20 points par cycle (moyen)")
    
    choix = input("\nVotre choix (1 ou 2): ")
    
    if choix == "1":
        nb_points = 5
    else:
        nb_points = 20
    
    envoyer_signal_mqtt(nb_points_echantillon=nb_points)