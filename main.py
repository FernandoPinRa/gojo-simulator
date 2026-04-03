import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc"
MODELO_MANOS = "hand_landmarker.task"

fuente = ImageFont.truetype(FONT_PATH, 120)

base_options = python.BaseOptions(model_asset_path=MODELO_MANOS)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

infinito = {"activa": False, "frame": 0}
purpura = {"activa": False, "frame": 0, "particulas": []}
estrellas = []


def iniciar(estado):
    estado["activa"] = True
    estado["frame"] = 0
    if "particulas" in estado:
        estado["particulas"] = []


def detener(estado):
    estado["activa"] = False
    estado["frame"] = 0
    if "particulas" in estado:
        estado["particulas"] = []


def distancia(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def detectar_punio(mano) -> bool:
    return (
        mano[8].y > mano[6].y and
        mano[12].y > mano[10].y and
        mano[16].y > mano[14].y and
        mano[20].y > mano[18].y
    )


def detectar_uno(mano) -> bool:
    return (
        mano[8].y < mano[6].y and
        mano[12].y > mano[10].y and
        mano[16].y > mano[14].y and
        mano[20].y > mano[18].y
    )


def detectar_dos(mano) -> bool:
    return (
        mano[8].y < mano[6].y and
        mano[12].y < mano[10].y and
        mano[16].y > mano[14].y and
        mano[20].y > mano[18].y
    )


def detectar_tres(mano) -> bool:
    return (
        mano[8].y < mano[6].y and
        mano[12].y < mano[10].y and
        mano[16].y < mano[14].y and
        mano[20].y > mano[18].y
    )


def detectar_dos_manos(manos) -> bool:
    return len(manos) == 2


def detectar_infinito(mano) -> bool:
    return (
        mano[8].y < mano[6].y and
        mano[12].y < mano[10].y and
        mano[16].y > mano[14].y and
        mano[20].y > mano[18].y and
        mano[8].x > mano[12].x
    )


def detectar_purpura(mano, umbral=0.5) -> bool:
    tamano_mano = distancia(mano[0], mano[9])
    limite = tamano_mano * umbral
    pulgar, indice, corazon = mano[4], mano[8], mano[12]
    dedos_juntos = (
        distancia(pulgar, indice) < limite and
        distancia(pulgar, corazon) < limite and
        distancia(indice, corazon) < limite
    )
    return dedos_juntos and (mano[16].y < mano[14].y or mano[20].y < mano[18].y)


def mezclar(frame, overlay, alpha):
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def oscurecer(frame, intensidad):
    return mezclar(frame, np.zeros_like(frame), intensidad / 255)


def dibujar_orbe(frame, cx, cy, radio, alpha, color_bgr):
    overlay = frame.copy()
    for r in range(radio, 0, -8):
        factor = (1 - r / radio) * alpha
        c = tuple(int(ch * factor) for ch in color_bgr)
        cv2.circle(overlay, (cx, cy), r, c, -1)
    cv2.circle(overlay, (cx, cy), max(1, radio // 6), (255, 255, 255), -1)
    return mezclar(frame, overlay, alpha * 0.7)


def dibujar_rayos(frame, cx, cy, n_rayos, longitud, alpha, rotacion=0.0):
    overlay = frame.copy()
    for i in range(n_rayos):
        angle = (2 * np.pi / n_rayos) * i + rotacion
        x2 = int(cx + np.cos(angle) * longitud)
        y2 = int(cy + np.sin(angle) * longitud)
        cv2.line(overlay, (cx, cy), (x2, y2), (255, 220, 180), 1)
    return mezclar(frame, overlay, alpha)


def dibujar_texto(frame, texto, x, y, alpha, color_rgb):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    r, g, b = [int(c * alpha) for c in color_rgb]
    draw.text((x, y), texto, font=fuente, fill=(r, g, b))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def generar_estrellas(w, h, n=120):
    rng = np.random.default_rng(42)
    return [
        {
            "x": int(rng.integers(0, w)),
            "y": int(rng.integers(0, h)),
            "radio": int(rng.integers(1, 3)),
            "fase": float(rng.uniform(0, 2 * np.pi)),
        }
        for _ in range(n)
    ]


def dibujar_estrellas(frame, estrellas, t):
    for e in estrellas:
        brillo = int(255 * (0.5 + 0.5 * np.sin(e["fase"] + t * 0.1)))
        cv2.circle(frame, (e["x"], e["y"]), e["radio"], (brillo, brillo, brillo), -1)


def procesar_animacion_infinito(frame, w, h):
    global estrellas

    infinito["frame"] += 1
    f = infinito["frame"]
    cx, cy = w // 2, h // 2

    if f <= 40:
        t = f / 40
        frame = oscurecer(frame, int(220 * t))
        if f <= 10:
            flash = 1 - f / 10
            frame = mezclar(frame, np.full_like(frame, 255), flash * 0.8)
        cv2.circle(frame, (cx, cy), int(t * max(w, h)), (180, 100, 50), 2)

    elif f <= 280:
        t = (f - 40) / 240
        frame = oscurecer(frame, 230)

        if not estrellas:
            estrellas = generar_estrellas(w, h)
        dibujar_estrellas(frame, estrellas, f)

        longitud = int(min(w, h) * 0.8 * min(t * 2, 1.0))
        frame = dibujar_rayos(frame, cx, cy, 80, longitud, min(t * 3, 0.4), f * 0.01)

        frame = dibujar_orbe(frame, cx, cy, 30 + int(np.sin(f * 0.15) * 8), 0.9, (255, 180, 80))

        for k in range(1, 4):
            radio_anillo = (f * 2 + k * 60) % max(w, h)
            a = max(0.0, 1 - radio_anillo / max(w, h))
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), int(radio_anillo), (200, 150, 80), 1)
            frame = mezclar(frame, overlay, a * 0.3)

        if f > 100:
            alpha_texto = min((f - 100) / 60, 1.0)
            frame = dibujar_texto(frame, "無量空処", w // 2 - 240, h // 2 - 80, alpha_texto, (180, 220, 255))

    else:
        t = (f - 280) / 20
        frame = mezclar(frame, np.full_like(frame, 255), t)

    if f > 300:
        detener(infinito)
        estrellas = []

    return frame


def nueva_particula_purpura(cx, cy):
    angle = random.uniform(0, 2 * np.pi)
    speed = random.uniform(4, 12)
    return {
        "x": float(cx), "y": float(cy),
        "vx": speed * np.cos(angle), "vy": speed * np.sin(angle),
        "radio": random.randint(2, 6),
        "vida": 0, "max_vida": random.randint(30, 80),
    }


def actualizar_particulas_purpura(frame, particulas):
    vivas = []
    for p in particulas:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vida"] += 1
        if p["vida"] < p["max_vida"]:
            a = 1 - p["vida"] / p["max_vida"]
            cv2.circle(frame, (int(p["x"]), int(p["y"])), p["radio"],
                       (int(255 * a), int(20 * a), int(210 * a)), -1)
            vivas.append(p)
    return vivas


def procesar_animacion_purpura(frame, w, h):
    purpura["frame"] += 1
    f = purpura["frame"]
    cx, cy = w // 2, h // 2

    if f <= 50:
        t = f / 50
        radio = int(20 + 30 * t)
        bx = int(w * 0.1 + (cx - w * 0.1) * t)
        rx = int(w * 0.9 - (w * 0.9 - cx) * t)
        for pos, color in [(bx, (255, 80, 0)), (rx, (0, 30, 255))]:
            overlay = frame.copy()
            cv2.circle(overlay, (pos, cy), radio, color, -1)
            frame = mezclar(frame, overlay, t * 0.6)

    elif f <= 110:
        t = f - 50
        radio = 70 + int(np.sin(t * 0.4) * 12)
        frame = dibujar_orbe(frame, cx, cy, radio, min(1.0, t / 20), (255, 0, 220))
        purpura["particulas"].append(nueva_particula_purpura(cx, cy))

    elif f <= 155:
        t = f - 110
        frame = dibujar_orbe(frame, cx, cy, int(70 + t * 10), max(0.0, 1 - t / 45), (255, 0, 220))
        flash = max(0.0, 0.9 - t / 30)
        frame = mezclar(frame, np.full_like(frame, 255), flash)
        for _ in range(6):
            purpura["particulas"].append(nueva_particula_purpura(cx, cy))

    elif f <= 230:
        t = f - 155
        if t > 20:
            alpha_texto = min((t - 20) / 40, 1.0)
            frame = dibujar_texto(frame, "虚式「紫」", w // 2 - 220, h // 2 - 80, alpha_texto, (255, 80, 255))

    purpura["particulas"] = actualizar_particulas_purpura(frame, purpura["particulas"])

    if f > 230:
        detener(purpura)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    gestos = {
        handedness[0].category_name: hand
        for hand, handedness in zip(result.hand_landmarks, result.handedness)
    }

    mano_izq = gestos.get("Left")
    mano_der = gestos.get("Right")

    if mano_izq and detectar_infinito(mano_izq) and not infinito["activa"]:
        iniciar(infinito)

    if mano_izq and detectar_punio(mano_izq) and infinito["activa"]:
        detener(infinito)
        estrellas = []

    if mano_izq and detectar_purpura(mano_izq) and not purpura["activa"]:
        iniciar(purpura)

    if mano_izq and detectar_punio(mano_izq) and purpura["activa"]:
        detener(purpura)

    if purpura["activa"]:
        frame = procesar_animacion_purpura(frame, w, h)

    if infinito["activa"]:
        frame = procesar_animacion_infinito(frame, w, h)

    cv2.imshow("Domain Expansion", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
