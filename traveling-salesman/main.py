"""
Distance matrix for a Traveling Salesman Problem:
Cities: K (Kathmandu), B (Bhaktapur), P (Patan), N (Nagarkot)

Distance Table (in km):
+-----+-----+-----+-----+-----+
| To  | K   | B   | P   | N   |
| From|     |     |     |     |
+-----+-----+-----+-----+-----+
| K   | 0   | 250 | 40  | 390 |
| B   | 250 | 0   | 100 | 200 |
| P   | 210 | 100 | 0   | 250 |
| N   | 390 | 200 | 250 | 0   |
+-----+-----+-----+-----+-----+
"""

# Initialize the distance matrix
distance_matrix = {
    'K': {'K': 0,   'B': 250, 'P': 40,  'N': 390},
    'B': {'K': 250, 'B': 0,   'P': 100, 'N': 200},
    'P': {'K': 210, 'B': 100, 'P': 0,   'N': 250},
    'N': {'K': 390, 'B': 200, 'P': 250, 'N': 0}
}

# City names for reference
cities = ['K', 'B', 'P', 'N']
city_names = {
    'K': 'Kathmandu',
    'B': 'Bhaktapur',
    'P': 'Patan',
    'N': 'Nagarkot'
}