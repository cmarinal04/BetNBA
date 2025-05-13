from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
import pandas as pd
import time

def get_games(season='2024-25', season_types=['Regular Season', 'Playoffs']):
    """
    Obtiene los juegos de la temporada especificada para los tipos de temporada dados.

    :param season: Temporada a consultar (por ejemplo, '2024-25').
    :param season_types: Lista de tipos de temporada (por ejemplo, ['Regular Season', 'Playoffs']).
    :return: DataFrame con los juegos combinados.
    """
    all_games = []
    for season_type in season_types:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable=season_type)
        games = gamefinder.get_data_frames()[0]
        games['SEASON_TYPE'] = season_type  # Agregar columna para identificar el tipo de temporada
        all_games.append(games)
    return pd.concat(all_games, ignore_index=True)

def get_player_stats(games, max_games=100):
    """
    Obtiene las estadísticas de los jugadores para los juegos especificados.

    :param games: DataFrame con los juegos.
    :param max_games: Número máximo de juegos a procesar.
    :return: DataFrame con las estadísticas de los jugadores.
    """
    stats_list = []

    for i, game_id in enumerate(games['GAME_ID'].unique()[:max_games]):
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            stats = boxscore.get_data_frames()[0]
            stats['GAME_ID'] = game_id
            stats_list.append(stats)
            print(f"Game {i+1} processed.")
            time.sleep(0.6)  # Evitar que la API te bloquee
        except Exception as e:
            print(f"Error en {game_id}: {e}")
            continue

    return pd.concat(stats_list, ignore_index=True)

# Paso 1: Obtener juegos (Regular Season y Playoffs)
games_df = get_games(season='2024-25', season_types=['Regular Season', 'Playoffs'])

# Paso 2: Obtener stats de jugadores (ajusta max_games según el tiempo/disponibilidad)
player_stats_df = get_player_stats(games_df, max_games=200)  # puedes subir a 1000+ después

# Guardar
player_stats_df.to_csv("nba_player_stats_2023.csv", index=False)