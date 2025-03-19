# Hand Tracking with MediaPipe

Questo progetto utilizza MediaPipe per il riconoscimento dei gesti della mano e OpenCV per la visualizzazione. Il codice è strutturato in modo che ogni parte sia una funzione separata, e include una classe per rappresentare i punti della mano.

## Librerie Utilizzate

- **OpenCV**: Utilizzata per la cattura video e la visualizzazione delle immagini.
- **MediaPipe**: Utilizzata per il riconoscimento dei gesti della mano.
- **PyAutoGUI**: Utilizzata per controllare il mouse in base ai movimenti della mano.
- **NumPy**: Utilizzata per operazioni matematiche e array.

## Funzionalità

- Riconoscimento dei gesti della mano.
- Disegno dei punti e delle connessioni della mano.
- Disegno di una linea tra la punta del pollice e la punta dell'indice.
- Controllo del mouse in base ai movimenti della mano.

## Come Usarlo

### Prerequisiti

Assicurati di avere installato le seguenti librerie:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

### Esecuzione del Codice

1. Clona il repository:

```bash
git clone https://github.com/CarloFanelli/handMouse.git
cd hand-tracking-mediapipe
```

2. Esegui lo script Python:

```bash
python hand_tracking.py
```
