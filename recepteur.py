import paho.mqtt.client as mqtt
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Configuration MQTT
BROKER = "localhost"
PORT = 1883
TOPIC = "ecg/signal"

# Stockage des données reçues
temps_recu = []
valeurs_recues = []
reception_terminee = False

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

def afficher_resultats():
    """Affiche les résultats de la réception"""
    print("\n" + "=" * 60)
    print("   ANALYSE DES RÉSULTATS")
    print("=" * 60)
    
    print(f"\n📊 Nombre de points reçus: {len(valeurs_recues)}")
    print(f"⏱️  Durée totale: {max(temps_recu):.2f} secondes")
    print(f"📉 Points par cycle: ~{len(valeurs_recues) // 5}")
    
    # Générer le signal original pour comparaison
    temps_original, signal_original = generer_signal_original()
    
    # Créer la figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Graphique 1: Signal original
    ax1.plot(temps_original, signal_original, 'b-', linewidth=1.5, label='Signal ECG Original')
    ax1.set_title('Signal ECG Original (Haute résolution)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Temps (s)', fontsize=12)
    ax1.set_ylabel('Amplitude (mV)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Graphique 2: Signal reçu (échantillonné)
    ax2.plot(temps_recu, valeurs_recues, 'ro-', linewidth=2, markersize=6, 
             label=f'Signal Reçu ({len(valeurs_recues)} points)')
    ax2.set_title(f'Signal ECG Reçu via MQTT (Échantillonné - ~{len(valeurs_recues)//5} points/cycle)', 
                  fontsize=14, fontweight='bold', color='darkred')
    ax2.set_xlabel('Temps (s)', fontsize=12)
    ax2.set_ylabel('Amplitude (mV)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Ajouter un titre général
    fig.suptitle('Comparaison: Signal Original vs Signal Échantillonné', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    plt.savefig(f'ecg_comparaison_{len(valeurs_recues)}_points.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Graphique sauvegardé: ecg_comparaison_{len(valeurs_recues)}_points.png")
    
    plt.show()
    
    # Analyse de la qualité
    print("\n📈 ANALYSE DE LA QUALITÉ:")
    if len(valeurs_recues) < 50:
        print("❌ FAIBLE: Perte importante de détails du signal")
        print("   - Les pics R ne sont pas bien définis")
        print("   - Les ondes P et T sont presque invisibles")
        print("   - Risque de ne pas détecter des anomalies cardiaques")
    elif len(valeurs_recues) < 150:
        print("⚠️  MOYENNE: Signal reconnaissable mais dégradé")
        print("   - Les pics R sont visibles")
        print("   - Les ondes P et T sont partiellement détectables")
        print("   - Détection d'anomalies limitée")
    else:
        print("✅ BONNE: Signal bien préservé")
        print("   - Toutes les composantes sont visibles")
        print("   - Détection d'anomalies possible")

def recevoir_signal():
    """Fonction principale de réception"""
    global reception_terminee
    
    client = mqtt.Client("ECG_Edge")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        print("=" * 60)
        print("   EDGE IoT - RÉCEPTEUR ECG")
        print("=" * 60)
        print("\n🔌 Connexion au broker MQTT...\n")
        
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        
        # Attendre la fin de la réception
        print("⏳ En attente des données...\n")
        while not reception_terminee:
            pass
        
        # Afficher les résultats
        if len(valeurs_recues) > 0:
            afficher_resultats()
        else:
            print("❌ Aucune donnée reçue")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("\n👋 Déconnexion du broker")

if __name__ == "__main__":
    recevoir_signal()