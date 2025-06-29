import requests
import pandas as pd
import json
import time
from datetime import datetime
import random
import numpy as np


class APIFootballClient:

  def __init__(self, api_key):
    self.api_key = api_key
    self.base_url = "https://v3.football.api-sports.io"
    self.headers = {'x-apisports-key': api_key}

  def get_premier_league_fixtures(self, season=2024):
    """Get Premier League fixtures for 2024/2025 season"""
    url = f"{self.base_url}/fixtures"
    params = {
        'league': 39,  # Premier League ID
        'season': season
    }

    response = requests.get(url, headers=self.headers, params=params)
    if response.status_code == 200:
      return response.json()
    else:
      print(f"Error: {response.status_code}")
      return None

  def get_match_events(self, fixture_id):
    """Get detailed events for a specific match"""
    url = f"{self.base_url}/fixtures/events"
    params = {'fixture': fixture_id}

    response = requests.get(url, headers=self.headers, params=params)
    if response.status_code == 200:
      return response.json()
    else:
      print(
          f"Error getting events for fixture {fixture_id}: {response.status_code}"
      )
      return None

  def get_match_lineups(self, fixture_id):
    """Get lineups for a specific match"""
    url = f"{self.base_url}/fixtures/lineups"
    params = {'fixture': fixture_id}

    response = requests.get(url, headers=self.headers, params=params)
    if response.status_code == 200:
      return response.json()
    else:
      print(
          f"Error getting lineups for fixture {fixture_id}: {response.status_code}"
      )
      return None

  def get_teams(self, league_id=39, season=2024):
    """Get Premier League teams"""
    url = f"{self.base_url}/teams"
    params = {'league': league_id, 'season': season}

    response = requests.get(url, headers=self.headers, params=params)
    if response.status_code == 200:
      return response.json()
    else:
      print(f"Error getting teams: {response.status_code}")
      return None
