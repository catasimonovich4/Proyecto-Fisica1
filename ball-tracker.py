import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
import os
import time

class BallTracker:
    def __init__(self, video_path, calibration_factor=1.0, ball_color="any"):
        """
        Inicializa el rastreador de la pelota.
        
        Args:
            video_path: Ruta al archivo de video
            calibration_factor: Factor de conversión de píxeles a metros
            ball_color: Color de la pelota para facilitar el seguimiento ("red", "blue", "yellow", "any")
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dt = 1/self.fps  # tiempo entre cuadros
        
        self.calibration_factor = calibration_factor  # metros por píxel
        self.g = 9.81  # aceleración debida a la gravedad en m/s²
        
        self.ball_color = ball_color
        self.positions = []
        self.times = []
        self.impulse_start_frame = None
        self.impulse_end_frame = None
        self.contact_loss_frame = None
        self.ground_impact_frame = None
        
        # Para la interfaz de selección manual
        self.roi_selector_active = False
        self.current_frame = None
        self.selecting_ball = False
        self.selection_start_pos = None
        self.selection_rect = None
        
    def _get_color_mask(self, frame_hsv):
        """Crea una máscara basada en el color elegido de la pelota"""
        if self.ball_color == "red":
            # Rojo en HSV (hay que manejar el wrap-around del rojo en HSV)
            mask1 = cv2.inRange(frame_hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(frame_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            return mask1 | mask2
        elif self.ball_color == "blue":
            return cv2.inRange(frame_hsv, np.array([100, 150, 50]), np.array([140, 255, 255]))
        elif self.ball_color == "yellow":
            return cv2.inRange(frame_hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        else:  # "any" - detectará basado en movimiento y forma
            return None

    def _detect_ball(self, frame, prev_frame=None, background=None):
        """
        Detecta la pelota en el cuadro actual usando color, movimiento o forma.
        Retorna las coordenadas (x, y) del centro de la pelota.
        """
        # Convertir a HSV para mejor detección de color
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # 1. Intenta detectar por color si se especificó
        if self.ball_color != "any":
            mask = self._get_color_mask(frame_hsv)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Encuentra contornos en la máscara
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Asume que el contorno más grande es la pelota
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                # Solo procede si el radio cumple con un tamaño mínimo
                if radius > 10:
                    return (int(x), int(y))
        
        # 2. Intenta detectar por movimiento si tenemos un cuadro previo
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_blurred = cv2.GaussianBlur(prev_gray, (11, 11), 0)
            
            # Diferencia absoluta entre cuadros
            frame_diff = cv2.absdiff(blurred, prev_blurred)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Mejora la máscara
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Encuentra contornos
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Filtra contornos por área y circularidad
                valid_contours = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 50:  # Ignora contornos muy pequeños
                        continue
                        
                    # Calcula la circularidad
                    perimeter = cv2.arcLength(c, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.6:  # Objetos circulares tienen valores cercanos a 1
                        valid_contours.append(c)
                
                if valid_contours:
                    # Usa el contorno más grande que es bastante circular
                    c = max(valid_contours, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    return (int(x), int(y))
        
        # 3. Detección basada en la transformada de Hough (detecta círculos)
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.5, 
            minDist=30,
            param1=100, 
            param2=30, 
            minRadius=10, 
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Toma el círculo más fuerte detectado
            x, y, _ = circles[0, 0]
            return (int(x), int(y))
            
        return None

    def select_roi_callback(self, event, x, y, flags, param):
        """Callback para la selección manual de la región de interés donde está la pelota"""
        if self.selecting_ball:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selection_start_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.selection_start_pos:
                self.selection_rect = (self.selection_start_pos[0], self.selection_start_pos[1], 
                                      x - self.selection_start_pos[0], y - self.selection_start_pos[1])
                img_copy = self.current_frame.copy()
                cv2.rectangle(img_copy, self.selection_start_pos, (x, y), (0, 255, 0), 2)
                cv2.imshow("Frame", img_copy)
            elif event == cv2.EVENT_LBUTTONUP:
                if abs(x - self.selection_start_pos[0]) > 5 and abs(y - self.selection_start_pos[1]) > 5:
                    self.selection_rect = (self.selection_start_pos[0], self.selection_start_pos[1], 
                                          x - self.selection_start_pos[0], y - self.selection_start_pos[1])
                    self.selecting_ball = False
                    self.roi_selector_active = False
                    cv2.destroyWindow("Frame")

    def manual_select_first_ball_position(self):
        """Permite al usuario seleccionar manualmente la posición inicial de la pelota"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, self.current_frame = self.cap.read()
        if not ret:
            return None
            
        self.selecting_ball = True
        self.roi_selector_active = True
        
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.select_roi_callback)
        
        while self.roi_selector_active and cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) >= 1:
            if self.selection_rect is None:
                cv2.imshow("Frame", self.current_frame)
            cv2.waitKey(1)
            
        if self.selection_rect:
            # Calcular el centro del rectángulo seleccionado
            x = self.selection_rect[0] + self.selection_rect[2] // 2
            y = self.selection_rect[1] + self.selection_rect[3] // 2
            return (x, y)
        return None
        
    def track_ball(self, show_video=True, output_path=None):
        """
        Rastrea la pelota a lo largo del video y guarda sus posiciones.
        
        Args:
            show_video: Si es True, muestra el video durante el procesamiento
            output_path: Ruta donde guardar el video procesado (opcional)
        """
        print("Iniciando seguimiento de la pelota...")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Pedir al usuario que seleccione la posición inicial de la pelota
        initial_position = self.manual_select_first_ball_position()
        if initial_position:
            self.positions.append(initial_position)
            self.times.append(0)
        
        frame_idx = 0
        prev_frame = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Guardar una copia para visualización
            display_frame = frame.copy()
            
            # Si ya tenemos una posición anterior, buscar cerca de ella
            ball_pos = None
            if self.positions and frame_idx > 0:
                # Predecir dónde debería estar la pelota basado en el movimiento anterior
                prev_pos = self.positions[-1]
                
                # Si tenemos al menos dos posiciones anteriores, predecir basado en velocidad
                if len(self.positions) >= 2:
                    prev_prev_pos = self.positions[-2]
                    dx = prev_pos[0] - prev_prev_pos[0]
                    dy = prev_pos[1] - prev_prev_pos[1]
                    
                    # Region of interest centrada en la posición predicha
                    search_center = (prev_pos[0] + dx, prev_pos[1] + dy)
                    search_radius = 100  # pixels
                    
                    # Recortar ROI para búsqueda
                    roi_x1 = max(0, int(search_center[0] - search_radius))
                    roi_y1 = max(0, int(search_center[1] - search_radius))
                    roi_x2 = min(self.width, int(search_center[0] + search_radius))
                    roi_y2 = min(self.height, int(search_center[1] + search_radius))
                    
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi.size > 0:
                        roi_prev = prev_frame[roi_y1:roi_y2, roi_x1:roi_x2] if prev_frame is not None else None
                        
                        ball_roi_pos = self._detect_ball(roi, roi_prev)
                        if ball_roi_pos:
                            ball_pos = (ball_roi_pos[0] + roi_x1, ball_roi_pos[1] + roi_y1)
            
            # Si no se encontró con la predicción, buscar en todo el frame
            if ball_pos is None:
                ball_pos = self._detect_ball(frame, prev_frame)
            
            # Si se detectó la pelota, guardar posición y tiempo
            if ball_pos:
                self.positions.append(ball_pos)
                self.times.append(frame_idx / self.fps)
                
                # Dibujar el centro de la pelota y su trayectoria
                cv2.circle(display_frame, ball_pos, 5, (0, 255, 0), -1)
                
                # Dibujar la trayectoria
                if len(self.positions) > 1:
                    for i in range(1, min(20, len(self.positions))):
                        if len(self.positions) - i < 0:
                            continue
                        pt1 = self.positions[-i]
                        pt2 = self.positions[-i-1]
                        cv2.line(display_frame, pt1, pt2, (0, 0, 255), 2)
                
                # Mostrar información sobre la posición
                text = f"Frame: {frame_idx}, Pos: ({ball_pos[0]}, {ball_pos[1]})"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 0), 2)
            
            if show_video:
                cv2.imshow('Tracking', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Marcar frame como inicio/fin del impulso
                    if self.impulse_start_frame is None:
                        self.impulse_start_frame = frame_idx
                        print(f"Inicio del impulso marcado en frame {frame_idx}")
                    elif self.impulse_end_frame is None:
                        self.impulse_end_frame = frame_idx
                        print(f"Fin del impulso marcado en frame {frame_idx}")
                elif key == ord('c'):  # Marcar frame como pérdida de contacto
                    self.contact_loss_frame = frame_idx
                    print(f"Pérdida de contacto marcada en frame {frame_idx}")
                elif key == ord('g'):  # Marcar frame como impacto con el suelo
                    self.ground_impact_frame = frame_idx
                    print(f"Impacto con el suelo marcado en frame {frame_idx}")
            
            if output_path:
                out.write(display_frame)
                
            prev_frame = frame.copy()
            frame_idx += 1
            
            # Mostrar progreso
            if frame_idx % 30 == 0:
                print(f"Procesando frame {frame_idx}/{self.total_frames} ({frame_idx/self.total_frames*100:.1f}%)")
        
        if output_path:
            out.release()
            
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"Seguimiento completado. Se detectaron {len(self.positions)} posiciones.")
        
        # Si no se marcaron frames importantes, intentar detectarlos automáticamente
        if self.impulse_start_frame is None:
            self.detect_key_frames()
            
        return self.positions, self.times
        
    def detect_key_frames(self):
        """Intenta detectar automáticamente frames clave como inicio/fin del impulso y rebote"""
        if len(self.positions) < 10:
            print("No hay suficientes posiciones para detectar frames clave.")
            return
            
        # Convertir posiciones a coordenadas y (invertidas para que y aumente hacia arriba)
        y_coords = [self.height - pos[1] for pos in self.positions]
        
        # Suavizar las posiciones para reducir ruido
        y_smooth = savgol_filter(y_coords, min(21, len(y_coords) - 1 if len(y_coords) % 2 == 0 else len(y_coords)), 3)
        
        # Calcular velocidades
        velocities = np.diff(y_smooth) / self.dt
        velocities = np.insert(velocities, 0, velocities[0])  # Repetir primer valor para mantener tamaño
        
        # Suavizar velocidades
        v_smooth = savgol_filter(velocities, min(21, len(velocities) - 1 if len(velocities) % 2 == 0 else len(velocities)), 3)
        
        # Calcular aceleraciones
        accelerations = np.diff(v_smooth) / self.dt
        accelerations = np.insert(accelerations, 0, accelerations[0])  # Repetir primer valor
        
        # Buscar cambios significativos en aceleración para detectar inicio/fin del impulso
        acc_threshold = np.std(accelerations) * 2
        
        # Inicio del impulso: primera aceleración significativa
        for i in range(10, len(accelerations)):
            if abs(accelerations[i]) > acc_threshold:
                self.impulse_start_frame = i - 5  # Ajustar un poco antes
                break
                
        # Fin del impulso / pérdida de contacto: cuando la aceleración vuelve a ser cercana a -g
        if self.impulse_start_frame is not None:
            for i in range(self.impulse_start_frame + 5, len(accelerations)):
                # Convertir aceleración a m/s² para comparar con g
                acc_ms2 = accelerations[i] * self.calibration_factor
                if abs(acc_ms2 + self.g) < self.g * 0.3:  # 30% de tolerancia
                    self.impulse_end_frame = i
                    self.contact_loss_frame = i
                    break
        
        # Impacto con el suelo: punto más bajo después del punto más alto
        max_height_idx = np.argmax(y_smooth)
        if max_height_idx < len(y_smooth) - 10:
            # Buscar el punto más bajo después del punto más alto
            for i in range(max_height_idx, len(y_smooth) - 1):
                if y_smooth[i] < y_smooth[i+1]:  # Cambio de dirección (rebote)
                    self.ground_impact_frame = i
                    break
        
        print(f"Detección automática: Impulso inicio={self.impulse_start_frame}, " +
              f"fin={self.impulse_end_frame}, impacto={self.ground_impact_frame}")
    
    def analyze_physics(self):
        """
        Analiza la física del movimiento, calculando velocidades, aceleraciones,
        energía, fuerza y otros parámetros físicos.
        """
        if len(self.positions) < 10:
            print("No hay suficientes datos para analizar.")
            return {}
        
        # Convertir posiciones a metros (y invertir el eje y para que hacia arriba sea positivo)
        positions_m = []
        for x, y in self.positions:
            # Convertir píxeles a metros y hacer que y aumente hacia arriba
            x_m = x * self.calibration_factor
            y_m = (self.height - y) * self.calibration_factor
            positions_m.append((x_m, y_m))
        
        # Extraer componentes x e y
        x_pos = [p[0] for p in positions_m]
        y_pos = [p[1] for p in positions_m]
        
        # Aplicar filtro Savitzky-Golay para suavizar las posiciones
        window = min(21, len(y_pos) - 1 if len(y_pos) % 2 == 0 else len(y_pos))
        poly = 3
        
        if len(y_pos) > window:
            y_smooth = savgol_filter(y_pos, window, poly)
            x_smooth = savgol_filter(x_pos, window, poly)
        else:
            y_smooth = y_pos
            x_smooth = x_pos
        
        # Calcular velocidades por diferenciación numérica
        vx = np.gradient(x_smooth, self.dt)
        vy = np.gradient(y_smooth, self.dt)
        
        # Magnitud de la velocidad
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Calcular aceleraciones
        ax = np.gradient(vx, self.dt)
        ay = np.gradient(vy, self.dt)
        
        # Magnitud de la aceleración
        a_mag = np.sqrt(ax**2 + ay**2)
        
        # Masa estimada de la pelota (asumimos 0.1 kg para una pelota pequeña)
        # Esto debería ser un parámetro configurable en un caso real
        mass = 0.1  # kg
        
        # Energía cinética: 1/2 * m * v²
        kinetic_energy = 0.5 * mass * v_mag**2
        
        # Energía potencial: m * g * h
        # Altura relativa al punto más bajo
        h_min = min(y_smooth)
        potential_energy = mass * self.g * (y_smooth - h_min)
        
        # Energía total
        total_energy = kinetic_energy + potential_energy
        
        # Fuerza aplicada: m * a
        force_x = mass * ax
        force_y = mass * ay
        force_mag = mass * a_mag
        
        # Analizar fase de impulso si se ha marcado
        impulse_data = {}
        if self.impulse_start_frame is not None and self.impulse_end_frame is not None:
            start_idx = self.impulse_start_frame
            end_idx = self.impulse_end_frame
            
            # Duración del impulso
            impulse_duration = (end_idx - start_idx) / self.fps
            
            # Velocidad inicial y final durante el impulso
            v_initial = v_mag[start_idx] if start_idx < len(v_mag) else 0
            v_final = v_mag[end_idx] if end_idx < len(v_mag) else v_mag[-1]
            
            # Aceleración promedio durante el impulso
            a_avg = (v_final - v_initial) / impulse_duration if impulse_duration > 0 else 0
            
            # Fuerza promedio durante el impulso
            f_avg = mass * a_avg
            
            # Impulso: F * Δt
            impulse = f_avg * impulse_duration
            
            # Altura máxima teórica: v²/(2g)
            max_height_theoretical = (v_final**2) / (2 * self.g)
            
            impulse_data = {
                "duration": impulse_duration,
                "velocity_initial": v_initial,
                "velocity_final": v_final,
                "acceleration_avg": a_avg,
                "force_avg": f_avg,
                "impulse": impulse,
                "max_height_theoretical": max_height_theoretical
            }
        
        # Analizar fase de vuelo libre
        flight_data = {}
        if self.contact_loss_frame is not None and self.ground_impact_frame is not None:
            start_idx = self.contact_loss_frame
            end_idx = self.ground_impact_frame
            
            # Duración del vuelo
            flight_duration = (end_idx - start_idx) / self.fps
            
            # Altura máxima real
            if start_idx < len(y_smooth) and end_idx < len(y_smooth):
                flight_y = y_smooth[start_idx:end_idx+1]
                max_height_idx = np.argmax(flight_y)
                max_height = flight_y[max_height_idx]
                
                # Posición y tiempo en el punto más alto
                max_height_time = self.times[start_idx + max_height_idx]
                
                # Velocidad en el punto más alto (debería ser cercana a cero en el eje y)
                max_height_vy = vy[start_idx + max_height_idx]
                
                flight_data = {
                    "duration": flight_duration,
                    "max_height": max_height,
                    "max_height_time": max_height_time,
                    "max_height_vy": max_height_vy
                }
                
                # Verificar conservación de energía
                if start_idx < len(total_energy) and start_idx + max_height_idx < len(total_energy):
                    energy_initial = total_energy[start_idx]
                    energy_at_max = total_energy[start_idx + max_height_idx]
                    energy_conservation = energy_at_max / energy_initial if energy_initial > 0 else 0
                    flight_data["energy_conservation"] = energy_conservation
        
        # Analizar rebote si está disponible
        bounce_data = {}
        if self.ground_impact_frame is not None and self.ground_impact_frame < len(self.positions) - 10:
            impact_idx = self.ground_impact_frame
            
            # Velocidad antes y después del impacto
            if impact_idx > 0 and impact_idx < len(v_mag) - 1:
                v_before = v_mag[impact_idx - 1]
                v_after = v_mag[impact_idx + 1]
                
                # Coeficiente de restitución: v_after / v_before
                coef_restitution = abs(v_after / v_before) if v_before > 0 else 0
                
                # Energía perdida en el impacto
                e_before = total_energy[impact_idx - 1]
                e_after = total_energy[impact_idx + 1]
                energy_loss = 1 - (e_after / e_before) if e_before > 0 else 0
                
                bounce_data = {
                    "velocity_before": v_before,
                    "velocity_after": v_after,
                    "coef_restitution": coef_restitution,
                    "energy_before": e_before,
                    "energy_after": e_after,
                    "energy_loss": energy_loss
                }
        
        # Empaquetar todos los datos para retornar
        results = {
            "times": self.times,
            "positions_x": x_smooth,
            "positions_y": y_smooth,
            "velocities_x": vx,
            "velocities_y": vy,
            "velocity_magnitude": v_mag,
            "accelerations_x": ax,
            "accelerations_y": ay,
            "acceleration_magnitude": a_mag,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": total_energy,
            "force_x": force_x,
            "force_y": force_y,
            "force_magnitude": force_mag,
            "impulse_phase": impulse_data,
            "flight_phase": flight_data,
            "bounce": bounce_data
        }
        
        return results
    
    def visualize_results(self, results, save_dir=None):
        """
        Visualiza los resultados del análisis físico con gráficos.
        
        Args:
            results: Resultados del análisis físico
            save_dir: Directorio donde guardar los gráficos generados
        """
        if not results:
            print("No hay resultados para visualizar.")
            return
            
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        times = results["times"]
        
        # Crear una figura con múltiples subgráficos
        plt.figure(figsize=(15, 20))
        
        # 1. Posición vs Tiempo
        plt.subplot(5, 1, 1)
        plt.plot(times, results["positions_y"], 'b-', label='Posición y')
        plt.plot(times, results["positions_x"], 'r-', label='Posición x')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Posición (m)')
        plt.title('Posición vs Tiempo')
        plt.grid(True)
        plt.legend()
        
        # Marcar fases si están disponibles
        if self.impulse_start_frame is not None and self.impulse_start_frame < len(times):
            plt.axvline(x=times[self.impulse_start_frame], color='g', linestyle='--', label='Inicio Impulso')
        if self.impulse_end_frame is not None and self.impulse_end_frame < len(times):
            plt.axvline(x=times[self.impulse_end_frame], color='r', linestyle='--', label='Fin Impulso')
        if self.ground_impact_frame is not None and self.ground_impact_frame < len(times):
            plt.axvline(x=times[self.ground_impact_frame], color='m', linestyle='--', label='Impacto Suelo')
        
        # 2. Velocidad vs Tiempo
        plt.subplot(5, 1, 2)
        plt.plot(times, results["velocities_y"], 'b-', label='Velocidad y')
        plt.plot(times, results["velocities_x"], 'r-', label='Velocidad x')
        plt.plot(times, results["velocity_magnitude"], 'k-', label='Magnitud')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Velocidad (m/s)')
        plt.title('Velocidad vs Tiempo')
        plt.grid(True)
        plt.legend()
        
        # Marcar fases
        if self.impulse_start_frame is not None and self.impulse_start_frame < len(times):
            plt.axvline(x=times[self.impulse_start_frame], color='g', linestyle='--')
        if self.impulse_end_frame is not None and self.impulse_end_frame < len(times):
            plt.axvline(x=times[self.impulse_end_frame], color='r', linestyle='--')
        if self.ground_impact_frame is not None and self.ground_impact_frame < len(times):
            plt.axvline(x=times[self.ground_impact_frame], color='m', linestyle='--')
        
        # 3. Aceleración vs Tiempo
        plt.subplot(5, 1, 3)
        plt.plot(times, results["accelerations_y"], 'b-', label='Aceleración y')
        plt.plot(times, results["accelerations_x"], 'r-', label='Aceleración x')
        plt.plot(times, results["acceleration_magnitude"], 'k-', label='Magnitud')
        # Línea horizontal para g
        plt.axhline(y=-self.g, color='g', linestyle='--', label='g = -9.81 m/s²')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Aceleración (m/s²)')
        plt.title('Aceleración vs Tiempo')
        plt.grid(True)
        plt.legend()
        
        # Marcar fases
        if self.impulse_start_frame is not None and self.impulse_start_frame < len(times):
            plt.axvline(x=times[self.impulse_start_frame], color='g', linestyle='--')
        if self.impulse_end_frame is not None and self.impulse_end_frame < len(times):
            plt.axvline(x=times[self.impulse_end_frame], color='r', linestyle='--')
        if self.ground_impact_frame is not None and self.ground_impact_frame < len(times):
            plt.axvline(x=times[self.ground_impact_frame], color='m', linestyle='--')
        
        # 4. Energía vs Tiempo
        plt.subplot(5, 1, 4)
        plt.plot(times, results["kinetic_energy"], 'r-', label='Energía Cinética')
        plt.plot(times, results["potential_energy"], 'b-', label='Energía Potencial')
        plt.plot(times, results["total_energy"], 'g-', label='Energía Total')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Energía (J)')
        plt.title('Energía vs Tiempo')
        plt.grid(True)
        plt.legend()
        
        # Marcar fases
        if self.impulse_start_frame is not None and self.impulse_start_frame < len(times):
            plt.axvline(x=times[self.impulse_start_frame], color='g', linestyle='--')
        if self.impulse_end_frame is not None and self.impulse_end_frame < len(times):
            plt.axvline(x=times[self.impulse_end_frame], color='r', linestyle='--')
        if self.ground_impact_frame is not None and self.ground_impact_frame < len(times):
            plt.axvline(x=times[self.ground_impact_frame], color='m', linestyle='--')
        
        # 5. Fuerza vs Tiempo
        plt.subplot(5, 1, 5)
        plt.plot(times, results["force_y"], 'b-', label='Fuerza y')
        plt.plot(times, results["force_x"], 'r-', label='Fuerza x')
        plt.plot(times, results["force_magnitude"], 'k-', label='Magnitud')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Fuerza (N)')
        plt.title('Fuerza vs Tiempo')
        plt.grid(True)
        plt.legend()
        
        # Marcar fases
        if self.impulse_start_frame is not None and self.impulse_start_frame < len(times):
            plt.axvline(x=times[self.impulse_start_frame], color='g', linestyle='--')
        if self.impulse_end_frame is not None and self.impulse_end_frame < len(times):
            plt.axvline(x=times[self.impulse_end_frame], color='r', linestyle='--')
        if self.ground_impact_frame is not None and self.ground_impact_frame < len(times):
            plt.axvline(x=times[self.ground_impact_frame], color='m', linestyle='--')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'physics_analysis.png'), dpi=300)
        
        plt.show()
        
        # Visualización de la trayectoria 2D
        plt.figure(figsize=(10, 8))
        plt.plot(results["positions_x"], results["positions_y"], 'b-', label='Trayectoria')
        
        # Marcar puntos clave
        if self.impulse_start_frame is not None and self.impulse_start_frame < len(results["positions_x"]):
            plt.plot(results["positions_x"][self.impulse_start_frame], 
                     results["positions_y"][self.impulse_start_frame], 
                     'go', markersize=8, label='Inicio Impulso')
        
        if self.impulse_end_frame is not None and self.impulse_end_frame < len(results["positions_x"]):
            plt.plot(results["positions_x"][self.impulse_end_frame], 
                     results["positions_y"][self.impulse_end_frame], 
                     'ro', markersize=8, label='Fin Impulso')
        
        if "flight_phase" in results and "max_height" in results["flight_phase"]:
            # Encontrar índice del punto más alto
            max_height_idx = np.argmax(results["positions_y"])
            plt.plot(results["positions_x"][max_height_idx], 
                     results["positions_y"][max_height_idx], 
                     'mo', markersize=8, label='Altura Máxima')
        
        if self.ground_impact_frame is not None and self.ground_impact_frame < len(results["positions_x"]):
            plt.plot(results["positions_x"][self.ground_impact_frame], 
                     results["positions_y"][self.ground_impact_frame], 
                     'ko', markersize=8, label='Impacto Suelo')
        
        plt.xlabel('Posición X (m)')
        plt.ylabel('Posición Y (m)')
        plt.title('Trayectoria de la Pelota')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Misma escala en ambos ejes
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'trayectoria.png'), dpi=300)
        
        plt.show()
        
        # Generar informe de texto con los resultados más importantes
        print("\n===== RESUMEN DEL ANÁLISIS FÍSICO =====")
        
        # Fase de impulso
        if "impulse_phase" in results and results["impulse_phase"]:
            impulse = results["impulse_phase"]
            print("\n--- FASE DE IMPULSO ---")
            print(f"Duración del impulso: {impulse['duration']:.3f} s")
            print(f"Velocidad inicial: {impulse['velocity_initial']:.2f} m/s")
            print(f"Velocidad final: {impulse['velocity_final']:.2f} m/s")
            print(f"Aceleración promedio: {impulse['acceleration_avg']:.2f} m/s²")
            print(f"Fuerza promedio aplicada: {impulse['force_avg']:.2f} N")
            print(f"Impulso total: {impulse['impulse']:.3f} Ns")
            print(f"Altura máxima teórica: {impulse['max_height_theoretical']:.2f} m")
        
        # Fase de vuelo libre
        if "flight_phase" in results and results["flight_phase"]:
            flight = results["flight_phase"]
            print("\n--- FASE DE VUELO LIBRE ---")
            print(f"Duración del vuelo: {flight['duration']:.3f} s")
            print(f"Altura máxima alcanzada: {flight['max_height']:.2f} m")
            print(f"Tiempo hasta punto más alto: {flight['max_height_time']:.3f} s")
            if "energy_conservation" in flight:
                print(f"Conservación de energía: {flight['energy_conservation']*100:.1f}%")
        
        # Rebote
        if "bounce" in results and results["bounce"]:
            bounce = results["bounce"]
            print("\n--- REBOTE ---")
            print(f"Velocidad antes del impacto: {bounce['velocity_before']:.2f} m/s")
            print(f"Velocidad después del impacto: {bounce['velocity_after']:.2f} m/s")
            print(f"Coeficiente de restitución: {bounce['coef_restitution']:.3f}")
            print(f"Energía perdida en el impacto: {bounce['energy_loss']*100:.1f}%")
        
        # Guardar informe en archivo de texto
        if save_dir:
            with open(os.path.join(save_dir, 'informe_fisico.txt'), 'w', encoding='utf-8') as f:
                f.write("===== RESUMEN DEL ANÁLISIS FÍSICO =====\n")
                
                # Fase de impulso
                if "impulse_phase" in results and results["impulse_phase"]:
                    impulse = results["impulse_phase"]
                    f.write("\n--- FASE DE IMPULSO ---\n")
                    f.write(f"Duración del impulso: {impulse['duration']:.3f} s\n")
                    f.write(f"Velocidad inicial: {impulse['velocity_initial']:.2f} m/s\n")
                    f.write(f"Velocidad final: {impulse['velocity_final']:.2f} m/s\n")
                    f.write(f"Aceleración promedio: {impulse['acceleration_avg']:.2f} m/s²\n")
                    f.write(f"Fuerza promedio aplicada: {impulse['force_avg']:.2f} N\n")
                    f.write(f"Impulso total: {impulse['impulse']:.3f} Ns\n")
                    f.write(f"Altura máxima teórica: {impulse['max_height_theoretical']:.2f} m\n")
                
                # Fase de vuelo libre
                if "flight_phase" in results and results["flight_phase"]:
                    flight = results["flight_phase"]
                    f.write("\n--- FASE DE VUELO LIBRE ---\n")
                    f.write(f"Duración del vuelo: {flight['duration']:.3f} s\n")
                    f.write(f"Altura máxima alcanzada: {flight['max_height']:.2f} m\n")
                    f.write(f"Tiempo hasta punto más alto: {flight['max_height_time']:.3f} s\n")
                    if "energy_conservation" in flight:
                        f.write(f"Conservación de energía: {flight['energy_conservation']*100:.1f}%\n")
                
                # Rebote
                if "bounce" in results and results["bounce"]:
                    bounce = results["bounce"]
                    f.write("\n--- REBOTE ---\n")
                    f.write(f"Velocidad antes del impacto: {bounce['velocity_before']:.2f} m/s\n")
                    f.write(f"Velocidad después del impacto: {bounce['velocity_after']:.2f} m/s\n")
                    f.write(f"Coeficiente de restitución: {bounce['coef_restitution']:.3f}\n")
                    f.write(f"Energía perdida en el impacto: {bounce['energy_loss']*100:.1f}%\n")
                    
                print(f"Informe guardado en {os.path.join(save_dir, 'informe_fisico.txt')}")


def calibration_tool(video_path):
    """
    Herramienta para calibrar el factor de conversión de píxeles a metros.
    El usuario selecciona un objeto de referencia de longitud conocida.
    
    Args:
        video_path: Ruta al archivo de video
    
    Returns:
        Factor de calibración (metros/píxel)
    """
    print("Herramienta de calibración iniciada.")
    print("Seleccione un objeto de referencia de longitud conocida en el video.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return 1.0
    
    # Leer el primer frame
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame del video.")
        cap.release()
        return 1.0
    
    # Variables para la selección
    ref_points = []
    selecting = True
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal ref_points, selecting, frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(ref_points) < 2:
                ref_points.append((x, y))
                img_copy = frame.copy()
                
                # Dibujar los puntos seleccionados
                for pt in ref_points:
                    cv2.circle(img_copy, pt, 5, (0, 255, 0), -1)
                
                # Si hay dos puntos, dibujar una línea entre ellos
                if len(ref_points) == 2:
                    cv2.line(img_copy, ref_points[0], ref_points[1], (0, 255, 0), 2)
                    # Calcular y mostrar la distancia en píxeles
                    dist_px = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + 
                                     (ref_points[1][1] - ref_points[0][1])**2)
                    text = f"Distancia: {dist_px:.1f} px"
                    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                
                cv2.imshow("Calibración", img_copy)
    
    # Crear ventana y establecer callback
    cv2.namedWindow("Calibración")
    cv2.setMouseCallback("Calibración", mouse_callback)
    
    # Mostrar instrucciones
    print("Haga clic en dos puntos para definir una distancia de referencia.")
    print("Presione ESC para salir.")
    
    cv2.imshow("Calibración", frame)
    
    # Esperar a que el usuario seleccione dos puntos o presione ESC
    while selecting and len(ref_points) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            selecting = False
    
    cv2.destroyAllWindows()
    cap.release()
    
    # Si se seleccionaron dos puntos, calcular el factor de calibración
    if len(ref_points) == 2:
        dist_px = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + 
                         (ref_points[1][1] - ref_points[0][1])**2)
        
        # Pedir al usuario la longitud real
        real_length = float(input("Ingrese la longitud real del objeto de referencia en metros: "))
        
        # Calcular el factor de calibración
        calib_factor = real_length / dist_px
        print(f"Factor de calibración: {calib_factor:.6f} metros/píxel")
        
        return calib_factor
    else:
        print("Calibración cancelada. Usando valor predeterminado.")
        return 1.0  # Valor predeterminado


def main():
    """Función principal del programa."""
    parser = argparse.ArgumentParser(description='Análisis de física del movimiento de una pelota.')
    parser.add_argument('video', help='Ruta al archivo de video')
    parser.add_argument('--calibrate', action='store_true', help='Ejecutar herramienta de calibración')
    parser.add_argument('--calib-factor', type=float, default=None, help='Factor de calibración (metros/píxel)')
    parser.add_argument('--ball-color', choices=['red', 'blue', 'yellow', 'any'], default='any',
                        help='Color de la pelota para facilitar el seguimiento')
    parser.add_argument('--output', default='output', help='Directorio de salida para resultados')
    parser.add_argument('--save-video', action='store_true', help='Guardar video con el seguimiento')
    parser.add_argument('--no-display', action='store_true', help='No mostrar video durante el procesamiento')
    
    args = parser.parse_args()
    
    # Verificar si el video existe
    if not os.path.exists(args.video):
        print(f"Error: El archivo de video {args.video} no existe.")
        return
    
    # Crear directorio de salida si no existe
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Calibración
    calib_factor = 0.01  # valor predeterminado (1 cm por píxel)
    if args.calibrate:
        calib_factor = calibration_tool(args.video)
    elif args.calib_factor is not None:
        calib_factor = args.calib_factor
    
    print(f"Usando factor de calibración: {calib_factor:.6f} metros/píxel")
    
    # Iniciar el seguimiento
    tracker = BallTracker(args.video, calib_factor, args.ball_color)
    
    # Nombre del archivo de video de salida
    output_video = None
    if args.save_video:
        video_name = os.path.basename(args.video)
        output_video = os.path.join(args.output, f"tracked_{video_name}")
        if not output_video.lower().endswith(('.mp4', '.avi')):
            output_video += '.avi'
    
    # Rastrear la pelota
    tracker.track_ball(not args.no_display, output_video)
    
    # Analizar la física
    results = tracker.analyze_physics()
    
    # Visualizar resultados
    tracker.visualize_results(results, args.output)
    
    print(f"Análisis completado. Resultados guardados en: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()