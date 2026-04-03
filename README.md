# Domain Expansion

Aplicación de visión por computadora en tiempo real inspirada en el anime **Jujutsu Kaisen**. Detecta gestos con las manos a través de la webcam y activa animaciones basadas en las técnicas de Gojo Satoru.

## Técnicas disponibles

| Gesto | Mano | Efecto |
|---|---|---|
| Índice y medio extendidos y cruzados | Izquierda | **無量空処** — Unlimited Void |
| Pulgar, índice y corazón juntos (anular o meñique extendido) | Izquierda | **虚式「紫」** — Hollow Purple |
| Puño cerrado | Izquierda | Detiene la animación activa |

## Requisitos

- Python 3.10+
- Webcam
- Fuente [Noto Sans CJK](https://github.com/notofonts/noto-cjk) instalada en el sistema
- Modelo `hand_landmarker.task` de MediaPipe ([descarga aquí](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker))

## Instalación

```bash
git clone https://github.com/tu-usuario/DomainExpansion.git
cd DomainExpansion

python -m venv .venv
source .venv/bin/activate

pip install opencv-python mediapipe Pillow numpy
```

Coloca el archivo `hand_landmarker.task` en la raíz del proyecto.

## Uso

```bash
source .venv/bin/activate
python main.py
```

Pulsa `q` para cerrar la aplicación.

## Estructura

```
DomainExpansion/
├── main.py                 # Aplicación principal
├── hand_landmarker.task    # Modelo de detección de manos (no incluido en el repo)
└── README.md
```

## .gitignore recomendado

Crea un `.gitignore` con lo siguiente para no subir el entorno virtual ni el modelo:

```
.venv/
hand_landmarker.task
__pycache__/
*.pyc
```
