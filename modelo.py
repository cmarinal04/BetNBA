# -------------------- Importaciones --------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time


# -------------------- Funciones de Preprocesamiento --------------------

def convert_minutes(min_str):
    """
    Convierte una cadena de tiempo en minutos a un valor flotante.
    Ejemplo: '35:27' -> 35.45, '33.000000' -> 33.0
    Si el valor ya es numérico, lo devuelve tal cual.
    """
    if pd.isnull(min_str):
        return None
    if isinstance(min_str, (int, float)):
        return float(min_str)
    if isinstance(min_str, str):
        if ':' in min_str:
            try:
                mins, secs = min_str.split(':')
                mins = float(mins.split('.')[0])  # Eliminar decimales en los minutos si existen
                return mins + int(secs) / 60
            except Exception:
                print(f"Error al convertir el valor '{min_str}' en minutos.")
                return None
        try:
            return float(min_str)
        except Exception:
            print(f"Error al convertir el valor '{min_str}' en flotante.")
            return None
    return None


def cargar_datos(filepath):
    """
    Carga el archivo CSV y realiza las conversiones iniciales.
    """
    df = pd.read_csv(filepath)
    df['MIN'] = df['MIN'].apply(convert_minutes)
    return df


def limpiar_datos(df, min_minutos=10):
    """
    Filtra y limpia los datos eliminando filas con pocos minutos jugados.
    """
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    df = df[df['MIN'] >= min_minutos].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def explorar_datos(df):
    """
    Muestra información básica del DataFrame.
    """
    print("Primeras filas del DataFrame:")
    print(df.head())
    print("\nColumnas disponibles:")
    print(df.columns.tolist())
    print("\nResumen estadístico:")
    print(df.describe())

# -------------------- Funciones para Partidos de Hoy --------------------

def obtener_partidos_hoy_con_selenium():
    """
    Obtiene los partidos de hoy usando Selenium para simular un navegador.
    """
    hoy = datetime.today().strftime('%Y-%m-%d')
    url = "https://www.espn.com/nba/schedule"

    # Configurar Selenium con Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")

    service = Service("D:/OneDrive - MIC/BI/Desarrollos/01 Desarrollo/Python/Proyecto/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)

    partidos = []
    rows = driver.find_elements(By.CSS_SELECTOR, "tbody.Table__TBODY tr.Table__TR--hasNote")
    for row in rows:
        equipos = row.find_elements(By.CSS_SELECTOR, "span.Table__Team a.AnchorLink")
        if len(equipos) == 2:
            equipo_local = equipos[1].text.strip()
            equipo_visitante = equipos[0].text.strip()
            hora = row.find_element(By.CSS_SELECTOR, "td.date__col a.AnchorLink").text.strip()
            partidos.append({'GAME_DATE': hoy, 'LOCAL': equipo_local, 'VISITANTE': equipo_visitante, 'TIME': hora})

    driver.quit()

    if partidos:
        return pd.DataFrame(partidos)
    else:
        print("No se encontraron partidos en la fuente web.")
        return pd.DataFrame()


def obtener_partidos_hoy_desde_sofascore():
    """
    Obtiene los partidos de hoy desde Sofascore usando requests y BeautifulSoup.

    :return: DataFrame con los partidos de hoy.
    """
    hoy = datetime.today().strftime('%Y-%m-%d')
    url = "https://www.sofascore.com/basketball"

    # Realizar la solicitud HTTP
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    # Verificar si la solicitud fue exitosa
    if response.status_code != 200:
        print(f"Error al acceder a la página web: {response.status_code}")
        return pd.DataFrame()

    # Analizar el contenido HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extraer los datos de los partidos
    partidos = []
    try:
        rows = soup.select("div.event-card")  # Selector para los eventos
        for row in rows:
            equipos = row.select("div.team-name")
            if len(equipos) == 2:
                equipo_local = equipos[0].text.strip()
                equipo_visitante = equipos[1].text.strip()
                hora = row.select_one("time").text.strip()
                partidos.append({'GAME_DATE': hoy, 'LOCAL': equipo_local, 'VISITANTE': equipo_visitante, 'TIME': hora})
    except Exception as e:
        print(f"Error al extraer los datos de Sofascore: {e}")

    if partidos:
        return pd.DataFrame(partidos)
    else:
        print("No se encontraron partidos en Sofascore.")
        return pd.DataFrame()

# -------------------- Funciones para Estadísticas --------------------


def obtener_boxscore_con_reintentos(game_id, max_reintentos=3, timeout=30):
    """
    Intenta obtener las estadísticas de un juego con reintentos en caso de error.

    :param game_id: ID del juego.
    :param max_reintentos: Número máximo de reintentos.
    :param timeout: Tiempo de espera para la solicitud.
    :return: DataFrame con las estadísticas del juego o un DataFrame vacío si falla.
    """
    for intento in range(max_reintentos):
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=timeout)
            return boxscore.get_data_frames()[0]
        except Exception as e:
            print(f"Error al obtener estadísticas para el juego {game_id} (intento {intento + 1}): {e}")
            time.sleep(5)  # Esperar 5 segundos antes de reintentar
    print(f"No se pudo obtener estadísticas para el juego {game_id} después de {max_reintentos} intentos.")
    return pd.DataFrame()


def obtener_estadisticas_historicas_por_equipo(equipos):
    """
    Obtiene las estadísticas históricas de los jugadores para los equipos especificados.

    :param equipos: Lista de equipos (abreviaturas).
    :return: Diccionario con las estadísticas históricas de los jugadores por equipo.
    """
    estadisticas_por_equipo = {}

    # Consultar todos los juegos de la temporada actual
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25', season_type_nullable='Playoffs')
        games = gamefinder.get_data_frames()[0]
    except Exception as e:
        print(f"Error al obtener datos de la API: {e}")
        return {}

    for equipo in equipos:
        print(f"\nObteniendo estadísticas históricas para el equipo: {equipo}")

        try:
            # Filtrar los juegos del equipo
            juegos_equipo = games[games['TEAM_ABBREVIATION'] == equipo]

            if juegos_equipo.empty:
                print(f"No se encontraron juegos para el equipo {equipo}.")
                estadisticas_por_equipo[equipo] = pd.DataFrame()
                continue

            # Obtener estadísticas de los jugadores en los juegos históricos
            stats_list = []
            for game_id in juegos_equipo['GAME_ID'].unique():
                try:
                    stats = obtener_boxscore_con_reintentos(game_id)
                    if not stats.empty:
                        stats['GAME_ID'] = game_id
                        stats_list.append(stats)
                except Exception as e:
                    print(f"Error al obtener estadísticas para el juego {game_id}: {e}")
                time.sleep(1)  # Esperar 1 segundo entre solicitudes

            if stats_list:
                # Combinar todas las estadísticas en un solo DataFrame
                estadisticas_por_equipo[equipo] = pd.concat(stats_list, ignore_index=True)
            else:
                print(f"No se encontraron estadísticas para el equipo {equipo}.")
                estadisticas_por_equipo[equipo] = pd.DataFrame()

        except Exception as e:
            print(f"Error al procesar los datos para el equipo {equipo}: {e}")
            estadisticas_por_equipo[equipo] = pd.DataFrame()

    return estadisticas_por_equipo


def calcular_metricas_por_equipo(estadisticas_por_equipo):
    """
    Calcula métricas relevantes (promedios) para los jugadores de cada equipo.

    :param estadisticas_por_equipo: Diccionario con las estadísticas históricas de los jugadores por equipo.
    :return: Diccionario con las métricas calculadas por equipo.
    """
    metricas_por_equipo = {}

    for equipo, stats in estadisticas_por_equipo.items():
        if not stats.empty:
            # Calcular promedios por jugador
            metricas = stats.groupby('PLAYER_NAME')[['PTS', 'REB', 'AST']].mean().reset_index()
            metricas_por_equipo[equipo] = metricas
        else:
            print(f"No hay estadísticas disponibles para el equipo {equipo}.")
            metricas_por_equipo[equipo] = pd.DataFrame()

    return metricas_por_equipo


def mostrar_metricas(metricas_por_equipo):
    """
    Muestra las métricas calculadas para cada equipo.

    :param metricas_por_equipo: Diccionario con las métricas calculadas por equipo.
    """
    for equipo, metricas in metricas_por_equipo.items():
        if not metricas.empty:
            print(f"\nMétricas para el equipo {equipo}:")
            print(metricas.to_string(index=False))
        else:
            print(f"No hay métricas disponibles para el equipo {equipo}.")

# -------------------- Función Principal --------------------

def obtener_top_5_por_jugador(estadisticas_por_equipo):
    """
    Obtiene las 5 mejores métricas (PTS, REB, AST) para cada jugador de cada equipo.

    :param estadisticas_por_equipo: Diccionario con las estadísticas históricas de los jugadores por equipo.
    :return: Diccionario con las 5 mejores métricas por jugador para cada equipo.
    """
    top_5_por_equipo = {}

    for equipo, stats in estadisticas_por_equipo.items():
        if not stats.empty:
            # Agrupar por jugador y calcular las 5 mejores métricas
            top_5_jugadores = stats.groupby('PLAYER_NAME')[['PTS', 'REB', 'AST']].apply(
                lambda x: x.nlargest(5, 'PTS')  # Ordenar por puntos y tomar las 5 mejores filas
            ).reset_index()
            top_5_por_equipo[equipo] = top_5_jugadores
        else:
            print(f"No hay estadísticas disponibles para el equipo {equipo}.")
            top_5_por_equipo[equipo] = pd.DataFrame()

    return top_5_por_equipo


def mostrar_top_5_por_equipo(top_5_por_equipo):
    """
    Muestra las 5 mejores métricas por jugador para cada equipo.

    :param top_5_por_equipo: Diccionario con las 5 mejores métricas por jugador para cada equipo.
    """
    for equipo, top_5 in top_5_por_equipo.items():
        if not top_5.empty:
            print(f"\nTop 5 métricas para el equipo {equipo}:")
            print(top_5.to_string(index=False))
        else:
            print(f"No hay métricas disponibles para el equipo {equipo}.")



def calcular_proyecciones_por_jugador(estadisticas_por_equipo):
    """
    Calcula las proyecciones promedio (PTS, REB, AST) para cada jugador basado en sus 5 mejores métricas.

    :param estadisticas_por_equipo: Diccionario con las estadísticas históricas de los jugadores por equipo.
    :return: Diccionario con las proyecciones promedio por jugador para cada equipo.
    """
    proyecciones_por_equipo = {}

    for equipo, stats in estadisticas_por_equipo.items():
        if not stats.empty:
            # Verificar las columnas disponibles en stats
            print(f"Columnas disponibles en stats para el equipo {equipo}: {stats.columns.tolist()}")

            if 'PLAYER_NAME' not in stats.columns:
                print(f"Error: La columna 'PLAYER_NAME' no está presente en las estadísticas del equipo {equipo}.")
                proyecciones_por_equipo[equipo] = pd.DataFrame()
                continue

            # Ordenar por puntos y tomar las 5 mejores métricas por jugador
            stats_sorted = stats.sort_values(by=['PLAYER_NAME', 'PTS'], ascending=[True, False])
            top_5_jugadores = stats_sorted.groupby('PLAYER_NAME').head(5)

            # Verificar las columnas disponibles en top_5_jugadores
            print(f"Columnas disponibles en top_5_jugadores para el equipo {equipo}: {top_5_jugadores.columns.tolist()}")

            if 'PLAYER_NAME' not in top_5_jugadores.columns:
                print(f"Error: La columna 'PLAYER_NAME' no está presente en top_5_jugadores del equipo {equipo}.")
                proyecciones_por_equipo[equipo] = pd.DataFrame()
                continue

            # Calcular promedios de las 5 mejores métricas por jugador
            proyecciones = top_5_jugadores.groupby('PLAYER_NAME')[['PTS', 'REB', 'AST']].mean().reset_index()
            proyecciones_por_equipo[equipo] = proyecciones
        else:
            print(f"No hay estadísticas disponibles para el equipo {equipo}.")
            proyecciones_por_equipo[equipo] = pd.DataFrame()

    return proyecciones_por_equipo


def mostrar_proyecciones_por_equipo(proyecciones_por_equipo):
    """
    Muestra las proyecciones promedio (PTS, REB, AST) por jugador para cada equipo.

    :param proyecciones_por_equipo: Diccionario con las proyecciones promedio por jugador para cada equipo.
    """
    for equipo, proyecciones in proyecciones_por_equipo.items():
        if not proyecciones.empty:
            print(f"\nProyecciones promedio para el equipo {equipo}:")
            print(proyecciones.to_string(index=False))
        else:
            print(f"No hay proyecciones disponibles para el equipo {equipo}.")


def convert_minutes(min_str):
    """
    Convierte una cadena de tiempo en minutos a un valor flotante.
    Ejemplo: '35:27' -> 35.45
    Si el valor ya es numérico, lo devuelve tal cual.
    """
    if isinstance(min_str, str) and ':' in min_str:
        try:
            # Manejar casos como '33.000000:07'
            mins, secs = min_str.split(':')
            mins = float(mins.split('.')[0])  # Eliminar decimales en los minutos
            return mins + int(secs) / 60
        except ValueError:
            print(f"Error al convertir el valor '{min_str}' en minutos.")
            return None
    try:
        return float(min_str)  # Si ya es un número, lo devuelve como flotante
    except (ValueError, TypeError):
        print(f"Error al convertir el valor '{min_str}' en flotante.")
        return None


def entrenar_modelo_para_metricas(stats, target):
    """
    Entrena un modelo de Random Forest para predecir una métrica específica (PTS, REB, AST).

    :param stats: DataFrame con las estadísticas históricas de los jugadores.
    :param target: La métrica objetivo a predecir (PTS, REB, AST).
    :return: Modelo entrenado.
    """
    # Seleccionar características relevantes
    features = ['MIN', 'FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'DREB', 'TO', 'STL', 'BLK']

    # Convertir la columna MIN a numérica (si es necesario)
    if 'MIN' in stats.columns:
        stats['MIN'] = stats['MIN'].apply(convert_minutes)

    # Imprimir datos antes de la limpieza
    print("\nDatos antes de la limpieza:")
    print(stats.head())

    # Eliminar filas con valores no válidos o nulos
    stats = stats.dropna(subset=features + [target])  # Eliminar filas con valores nulos
    for feature in features:
        stats[feature] = pd.to_numeric(stats[feature], errors='coerce')  # Convertir a numérico
    stats = stats.dropna(subset=features)  # Eliminar filas con valores no convertibles

    # Imprimir datos después de la limpieza
    print("\nDatos después de la limpieza:")
    print(stats.head())

    # Verificar que no haya valores no numéricos
    print(f"Datos después de limpieza para {target}:")
    print(stats[features + [target]].head())

    if stats.empty:
        print(f"Error: No hay datos suficientes para entrenar el modelo para {target}.")
        return None

    X = stats[features]
    y = stats[target]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModelo para {target}:")
    print(f" - R² Score: {r2:.4f}")
    print(f" - Error Cuadrático Medio (MSE): {mse:.2f}")

    return model


def proyectar_estadisticas_para_hoy(estadisticas_por_equipo, modelos):
    """
    Proyecta las estadísticas de los jugadores para los partidos de hoy.

    :param estadisticas_por_equipo: Diccionario con las estadísticas históricas de los jugadores por equipo.
    :param modelos: Diccionario con los modelos entrenados para PTS, REB y AST.
    :return: DataFrame con las proyecciones para los jugadores.
    """
    proyecciones = []

    for equipo, stats in estadisticas_por_equipo.items():
        print(f"\nProcesando estadísticas para el equipo {equipo}...")

        if not stats.empty:
            # Verificar si PLAYER_NAME está presente
            if 'PLAYER_NAME' not in stats.columns:
                print(f"Error: La columna 'PLAYER_NAME' no está presente en las estadísticas del equipo {equipo}.")
                continue

            # Imprimir una muestra de los datos originales
            print(f"Datos originales para el equipo {equipo}:")
            print(stats.head())

            # Eliminar filas con valores nulos en PLAYER_NAME
            stats = stats.dropna(subset=['PLAYER_NAME'])

            # Convertir la columna MIN usando convert_minutes
            stats['MIN'] = stats['MIN'].apply(convert_minutes)

            # Verificar si quedan datos después de eliminar nulos en PLAYER_NAME
            if stats.empty:
                print(f"No hay datos válidos en PLAYER_NAME para el equipo {equipo}.")
                continue

            # Asegurarse de que las columnas sean numéricas
            columnas_relevantes = ['MIN', 'FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'DREB', 'TO', 'STL', 'BLK']
            for col in columnas_relevantes:
                stats[col] = pd.to_numeric(stats[col], errors='coerce')

            # Imprimir datos después de convertir a numérico
            print(f"Datos después de convertir a numérico para el equipo {equipo}:")
            print(stats[columnas_relevantes].head())

            # Eliminar filas con valores no numéricos
            stats = stats.dropna(subset=columnas_relevantes)

            # Verificar si quedan datos después de eliminar valores no numéricos
            if stats.empty:
                print(f"No hay datos válidos en las columnas relevantes para el equipo {equipo}.")
                continue

            # Calcular las estadísticas promedio de cada jugador
            stats_promedio = stats.groupby('PLAYER_NAME')[columnas_relevantes].mean().reset_index()

            # Verificar si quedan jugadores después del agrupamiento
            if stats_promedio.empty:
                print(f"No se pudieron calcular promedios para el equipo {equipo}.")
                continue

            # Depurar: Imprimir los nombres de los jugadores procesados
            print(f"Jugadores procesados para el equipo {equipo}:")
            print(stats_promedio['PLAYER_NAME'].tolist())

            for _, jugador in stats_promedio.iterrows():
                entrada = jugador[columnas_relevantes].values.reshape(1, -1)

                # Proyectar puntos, rebotes y asistencias
                pts = modelos['PTS'].predict(entrada)[0]
                reb = modelos['REB'].predict(entrada)[0]
                ast = modelos['AST'].predict(entrada)[0]

                proyecciones.append({
                    'PLAYER_NAME': jugador['PLAYER_NAME'],
                    'TEAM': equipo,
                    'PTS_PROY': round(pts, 2),
                    'REB_PROY': round(reb, 2),
                    'AST_PROY': round(ast, 2)
                })
        else:
            print(f"No hay estadísticas disponibles para el equipo {equipo}.")

    return pd.DataFrame(proyecciones)


def main():
    # Equipos para los partidos de mañana
    equipos = ['IND', 'CLE', 'DEN', 'OKC']  # Indiana Pacers, Cleveland Cavaliers, Denver Nuggets, Oklahoma City Thunder

    # Obtener estadísticas históricas por equipo
    estadisticas_por_equipo = obtener_estadisticas_historicas_por_equipo(equipos)

    # Entrenar modelos para PTS, REB y AST
    stats_historicas = pd.concat(estadisticas_por_equipo.values(), ignore_index=True)
    modelos = {
        'PTS': entrenar_modelo_para_metricas(stats_historicas, 'PTS'),
        'REB': entrenar_modelo_para_metricas(stats_historicas, 'REB'),
        'AST': entrenar_modelo_para_metricas(stats_historicas, 'AST')
    }

    # Proyectar estadísticas para los partidos de mañana
    proyecciones = proyectar_estadisticas_para_hoy(estadisticas_por_equipo, modelos)

    # Verificar si el DataFrame de proyecciones está vacío
    if proyecciones.empty:
        print("\nNo se generaron proyecciones. Verifica los datos de entrada y el flujo de procesamiento.")
    else:
        # Mostrar las proyecciones
        print("\nProyecciones para los partidos de mañana:")
        print(proyecciones.to_string(index=False))

        # Guardar las proyecciones en un archivo Excel
        output_file = "proyecciones_partidos_manana.xlsx"
        proyecciones.to_excel(output_file, index=False)
        print(f"\nProyecciones guardadas en el archivo: {output_file}")

if __name__ == "__main__":
    main()