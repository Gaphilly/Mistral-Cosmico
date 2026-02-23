import unittest
from unittest.mock import patch

import backend


class ClimateStatsRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = backend.app.test_client()

    def test_missing_parameters_returns_400(self):
        response = self.client.get('/climate_stats?day=1&month=2')

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {'error': 'Missing parameters. Provide day, month, lat, lon.'},
        )

    @patch('backend.compute_wind_speed_stats', return_value=42)
    @patch('backend.compute_historical_stats', return_value=(-10.0, 55, 20))
    def test_cold_temperature_uses_snow_hail_key(self, _hist, _wind):
        response = self.client.get('/climate_stats?day=1&month=2&lat=45&lon=2')

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['snow_hail_freq_percent'], 55)
        self.assertNotIn('rainfall_gt_2mm_freq_percent', data)
        self.assertEqual(data['strong_winds_freq_percent'], 42)

    @patch('backend.compute_wind_speed_stats', return_value=8)
    @patch('backend.compute_historical_stats', return_value=(20.0, 25, 5))
    def test_mild_temperature_uses_rain_key(self, _hist, _wind):
        response = self.client.get('/climate_stats?day=1&month=2&lat=45&lon=2')

        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['rainfall_gt_2mm_freq_percent'], 25)
        self.assertNotIn('snow_hail_freq_percent', data)
        self.assertEqual(data['strong_winds_freq_percent'], 8)


if __name__ == '__main__':
    unittest.main()
